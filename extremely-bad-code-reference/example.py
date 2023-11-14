import semantic_kernel as sk
#from semantic_kernel.core_skills import TimeSkill
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_skills import ConversationSummarySkill

async def main():
    # Initialize the kernel
    kernel = sk.Kernel()

    # Configure LLM service
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_chat_service("chat_completion", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

    plugins_directory = "./plugins"

    # Import the OrchestratorPlugin from the plugins directory.
    orchestrator_plugin = kernel.import_semantic_skill_from_directory(
        plugins_directory, "OrchestratorPlugin"
    )
    conversation_summary_plugin = kernel.import_skill(
        ConversationSummarySkill(kernel=kernel), skill_name="ConversationSummarySkill"
    )

    # Create a new context and set the input, history, and options variables.
    context = kernel.create_new_context()
    context["input"] = "Yes"
    context[
        "history"
    ] = """Bot: How can I help you?
    User: What's the weather like today?
    Bot: Where are you located?
    User: I'm in Seattle.
    Bot: It's 70 degrees and sunny in Seattle today.
    User: Thanks! I'll wear shorts.
    Bot: You're welcome.
    User: Could you remind me what I have on my calendar today?
    Bot: You have a meeting with your team at 2:00 PM.
    User: Oh right! My team just hit a major milestone; I should send them an email to congratulate them.
    Bot: Would you like to write one for you?"""
    context["options"] = "SendEmail, ReadEmail, SendMeeting, RsvpToMeeting, SendChat"

    # Run the GetIntent function with the context.
    result = await kernel.run_async(
        orchestrator_plugin["GetIntent"],
        input_context=context,
    )

    print(result)


# Run the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())