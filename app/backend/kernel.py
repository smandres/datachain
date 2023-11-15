import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    OpenAITextEmbedding,
    OpenAIChatCompletion,
)
# Initialize the kernel
kernel = sk.Kernel()

# Configure LLM service
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("chat_completion", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

# Configure embeddings
kernel.add_text_embedding_generation_service("ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))

# Configure memory
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())

# Path to functions
plugins_directory = "./plugins"

# Get plugins
extract_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "Extract")
interpret_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "Extract")

# Get functions
analyzeDf = extract_plugin['analyzeDataframe']
generateQuestions = extract_plugin['generateQuestions']
decipherPrompt = interpret_plugin['decipherPrompt']

