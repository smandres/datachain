import semantic_kernel as sk
#from semantic_kernel.core_skills import TimeSkill
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planning import SequentialPlanner
#from semantic_kernel.core_skills import ConversationSummarySkill

async def main(prompt):
    # Initialize the kernel
    kernel = sk.Kernel()

    # Configure LLM service
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_chat_service("chat_completion", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

    plugins_directory = "./plugins"

    # Import the OrchestratorPlugin from the plugins directory.
    data_scientist = kernel.import_semantic_skill_from_directory(
        plugins_directory, "DataScientistPlugin"
    )

    # conversation_summary_plugin = kernel.import_skill(
    #     ConversationSummarySkill(kernel=kernel), skill_name="ConversationSummarySkill"
    # )

    #Create a new context and set the input, history, and options variables.
    my_context = kernel.create_new_context()
    my_context["prompt"] = prompt
    my_context["columns"] = "State, Deaths, Cases, FIPS, Date"
    my_context["descriptions"
    ] = """State: String of state in acroynm form i.e. NY
    Deaths: Integer of deaths
    Cases: Integer of Cases
    FIPS: Integer FIP code for the State
    Date: The date in form DD/MM/YYYY"""
    
#     @sk_function(
#     description="Extracts numbers from JSON",
#     name="ExtractNumbersFromJson",
# )
# def extract_numbers_from_json(self, context: SKContext):
#     numbers = json.loads(context["input"])

#     # Loop through numbers and add them to the context
#     for key, value in numbers.items():
#         if key == "number1":
#             # Add the first number to the input variable
#             context["input"] = str(value)
#         else:
#             # Add the rest of the numbers to the context
#             context[key] = str(value)

#     return context   
    # Create context
    # context = kernel.create_new_context()
    # context["prompt"] = prompt
    # context["metadata"] = metadata
                    
    # Run the GetIntent function with the context.

    # result = await kernel.run_async(
    #      data_scientist["QueryData"],
    #      data_scientist["DataUnderstanding"],
    #      data_scientist
    #      input_context=context,
    # )

    #print(result)

    # return result
    # Create planner

    planner = SequentialPlanner(kernel)

    ask = prompt
    sequential_plan = await planner.create_plan_async(goal=ask)

    plan_steps = ""
    for step in sequential_plan._steps:
        print(step.description, ":", step._state.__dict__)
        plan_steps += f"{step.description} : {step._state.__dict__}\n"

    result = await sequential_plan.invoke_async(context=my_context)

    print("\nPlan results:")
    print(result)
    
    return plan_steps, result

    


# # Run the main function
# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main())