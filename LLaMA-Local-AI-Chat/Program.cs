using LLama.Common;
using LLama;
//using Microsoft.VisualBasic;

using Microsoft.KernelMemory.AI;
using Microsoft.KernelMemory.AI.LlamaSharp;

using LLamaSharp.KernelMemory;
using Microsoft.KernelMemory.Configuration;
using Microsoft.KernelMemory;
using System.Diagnostics;
internal class Program
{
    private static async Task Main(string[] args)
    {
        try
        {
            // Indicate where the GGUF model file is
            // string modelPath = @"D:\models\llama-2-7b-chat.Q5_K_M.gguf";
            string modelPath = @"D:\models\Llama-2-7b-chat-hf-finetune-q5_k_m-v1.0.gguf";

            LLama.Common.InferenceParams infParams = new() { AntiPrompts = ["\n\n"] };
            LLamaSharpConfig lsConfig = new(modelPath) { DefaultInferenceParams = infParams };
            SearchClientConfig searchClientConfig = new() { MaxMatchesCount = 1, AnswerTokens = 50 };
            TextPartitioningOptions parseOptions = new() { MaxTokensPerParagraph = 300, MaxTokensPerLine = 100, OverlappingTokens = 30 };
            IKernelMemory memory = new KernelMemoryBuilder()
                .WithLLamaSharpDefaults(lsConfig)
                .WithSearchClientConfig(searchClientConfig)
                .With(parseOptions)
                .Build();

            // Ingest documents (format is automatically detected from the filename)
            string documentFolder = @"D:\models\documents";
            string[] documentPaths = Directory.GetFiles(documentFolder, "*.txt");
            for (int i = 0; i < documentPaths.Length; i++)
            {
                await memory.ImportDocumentAsync(documentPaths[i], steps: Constants.PipelineWithoutSummary);
            }

            // Allow the user to ask questions forever
            while (true)
            {
                Console.Write("\nQuestion: ");
                string question = Console.ReadLine() ?? string.Empty;
                MemoryAnswer answer = await memory.AskAsync(question);
                Console.WriteLine($"Answer: {answer.Result}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
        }
        
    }
}