from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import json
import time

def initial_chain(prompt, dataset):

    # Configure LLM service
    load_dotenv()
    api_key, org_id = os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_ORG_ID')
    client = OpenAI()

    # neat json printer
    def show_json(obj):
        temp = json.loads(obj.model_dump_json())
        pretty_json = json.dumps(temp, indent=4)
        print(pretty_json)

    # Create 'images' directory if it doesn't exist
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # create file
    # file = client.files.create(
    # file = open(r"C:\Users\cadet_admin\Desktop\pair-programming\datachain\app\Mall_Customers.csv", "rb"),
    # purpose='assistants'
    # )]
    path = r"C:\Users\cadet_admin\Desktop\pair-programming\datachain\app\data"
    new_path = path + "\\" + dataset
    # create file
    file = client.files.create(
    file = open(new_path, "rb"),
    purpose='assistants'
    )
    
    # create assistant here but I have one so I will use my assistant
    #assistant =  "asst_6Ug6p8RqTMMNZaXVAgtKUDnK"
    assistant = "asst_FB7tR3KmVEoKT0gZfN8gy0S0"
    #print("ASSISTANT")
    #show_json(assistant)

    # create thread
    thread = client.beta.threads.create()
    #print("THREAD")
    #show_json(thread)

    # create prompt #TODO: THIS IS WHERE USER PROMPT GOES
    message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user", 
    # """Do K-means and cluster the supermarket data.
    #  Spending Score is something I assign to the customer based on their defined parameters
    #   like customer behavior and purchasing data. I own the mall and want to understand the 
    #   customers like who can be easily converge [Target Customers] so that the sense can be 
    #   given to marketing team and plan the strategy accordingly.""",#,
    content= prompt,
    file_ids= [file.id] #TODO: this is where custom file goes
    )
    
    #print("USER MESSAGE")
    #show_json(message)

    # prepare thread for running
    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id= "asst_6Ug6p8RqTMMNZaXVAgtKUDnK",
    # model="gpt-4-1106-preview",
    # instructions="additional instructions",
    # tools=[{"type": "code_interpreter"}, {"type": "retrieval"}]
    )

    #show_json(run)

    # async function
    def wait_on_run(run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                # run thread
                thread_id=thread.id,
                run_id=run.id,
            )

            time.sleep(0.5)
        return run

    # run thread
    run = wait_on_run(run, thread)

    # get run steps
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread.id,
        run_id=run.id
    )  
    
    code_interpret = []
    # Check and print the step details
    for step in run_steps.data:
        code_obj = {}
        # if tool used
        if step.type == 'tool_calls':
            # Extract input and output
            input_value = step.step_details.tool_calls[0].code_interpreter.input
            output_value = step.step_details.tool_calls[0].code_interpreter.outputs

            # place into object
            code_obj["input"] = input_value
            code_obj["output"] = output_value
            code_obj["step_id"] = step.id
            code_obj["thread_id"] = step.thread_id
            code_obj["run_id"] = step.run_id

            # append to list
            code_interpret.append(code_obj)

            # # Print the input and output values for this step
            # print(f"Python Code Input: {input_value}")
            # print("\n")
            # print(f"Python Code Output: {output_value}")

    # # iterate through each message
    # for code in code_interpret:  
    #     print(f'STEP_ID: "{code["step_id"]}"')
    #     print(f'RUN_ID: "{code["run_id"]}"')
    #     print(f'THREAD_ID: "{code["thread_id"]}"')
    #     print(f'INPUT: "{code["input"]}"')  
    #     print(f'OUTPUT: "{code["output"]}"')    

    #print("RUN")
    #show_json(run)

    # get messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    #print("MESSAGES")
    #show_json(messages)

    # make into json
    messages = json.loads(messages.model_dump_json())
    
    # process each message
    processed_messages = []

    for message in messages["data"]:
        # Initialize new message object
        message_obj = {}

        message_obj["step_id"] = message["id"]
        message_obj["run_id"] = message["run_id"]
        message_obj["thread_id"] = message["thread_id"]
        message_obj["role"]  = message["role"]

        # Extract the text value and image file_id (if available)
        for content in message["content"]:
            if content["type"] == "text":
                message_obj["value"] = content["text"]["value"]
            elif content["type"] == "image_file" and "image_file" in content:
                message_obj["image_file_file_id"] = content["image_file"]["file_id"]

        processed_messages.append(message_obj)

    # convert to JSON
    #processed_messages_json = json.dumps(processed_messages, indent=4)
    #print("CLEAN RESPONSES")
    #print(processed_messages_json)

    # iterate through each message
    for message in processed_messages:
        if "image_file_file_id" in message:
            # access png if it exists
            image_file_id = message["image_file_file_id"]
            image_data = client.files.content(image_file_id)  # Replace 'image_file_id' with actual method to fetch file
            image_data_bytes = image_data.read()

            # save to images folder
            image_file_path = os.path.join(images_dir, f"{image_file_id}.png")
            with open(image_file_path, "wb") as file:
                file.write(image_data_bytes)

            # add this path as a key
            message["IMAGE"] = image_file_path
    
    for code in code_interpret:
        for message in processed_messages:
            if code["step_id"] == message["step_id"]:
                message["input"] = code["input"]
            else: 
                message["input"] = None

    #processed_messages_json = json.dumps(processed_messages, indent=4)
    #print("FULL")
    #print(processed_messages_json)

    #print(f'OUTPUT: "{code["output"]}"')    
    # # User messages first
    # for message in processed_messages:
    #     if message["role"] == "user":
    #         print(f'USER: "{message["value"]}"')
    #         print("\n")
    # # Then, print out assistant messages (start loop again bc its in order)
    # for message in processed_messages:
    #     if message["role"] != "user":
    #         print(f'ASSISTANT: "{message["value"]}"')
    #         print("\n")

    return processed_messages, thread.id, assistant

def recreate(prompt, thread, assistant):
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    # Configure LLM service
    load_dotenv()
    api_key, org_id = os.getenv('OPENAI_API_KEY'), os.getenv('OPENAI_ORG_ID')
    client = OpenAI()

    # Create a message to append to our thread
    message = client.beta.threads.messages.create(
         thread_id=thread, role="user", content=prompt)


    # Execute our run
    run = client.beta.threads.runs.create(
         thread_id=thread,
         assistant_id=assistant,
    )

    def wait_on_run(run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                # run thread
                thread_id=thread,
                run_id=run.id,
            )

            time.sleep(0.5)
        return run

    # Wait for completion
    wait_on_run(run, thread)

    # Retrieve all the messages added after our last user message
    messages = client.beta.threads.messages.list(
         thread_id=thread, order="asc", after=message.id
    )

    messages = json.loads(messages.model_dump_json())

    # process each message
    processed_messages = []

    for message in messages["data"]:
        # Initialize new message object
        message_obj = {}

        message_obj["step_id"] = message["id"]
        message_obj["run_id"] = message["run_id"]
        message_obj["thread_id"] = message["thread_id"]
        message_obj["role"]  = message["role"]

        # Extract the text value and image file_id (if available)
        for content in message["content"]:
            if content["type"] == "text":
                message_obj["value"] = content["text"]["value"]
            elif content["type"] == "image_file" and "image_file" in content:
                message_obj["image_file_file_id"] = content["image_file"]["file_id"]

        processed_messages.append(message_obj)

    # convert to JSON
    #processed_messages_json = json.dumps(processed_messages, indent=4)
    #print("CLEAN RESPONSES")
    #print(processed_messages_json)

    # iterate through each message
    for message in processed_messages:
        if "image_file_file_id" in message:
            # access png if it exists
            image_file_id = message["image_file_file_id"]
            image_data = client.files.content(image_file_id)  # Replace 'image_file_id' with actual method to fetch file
            image_data_bytes = image_data.read()

            # save to images folder
            image_file_path = os.path.join(images_dir, f"{image_file_id}.png")
            with open(image_file_path, "wb") as file:
                file.write(image_data_bytes)

            # add this path as a key
            message["IMAGE"] = image_file_path

    # print("UPDATED MESSAGE")
    # show_json(messages)

    # cleanup
    #response = client.beta.threads.delete("thread_abc123")
    #print(response)

    return processed_messages


# Run the main function
# if __name__ == "__main__":
#     main()