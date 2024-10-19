package devoxx.rag._1_introduction;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.output.Response;
import devoxx.rag.AbstractDevoxxTest;
import lombok.RequiredArgsConstructor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

class _10_chat_language_model extends AbstractDevoxxTest {

    @Test
    public void should_chat_language_model() {
        System.out.println(yellow("Using Chat Model:"));

        ChatLanguageModel chatModel = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("gpt-4o")
                .build();

        String question = "What is the sky bue ?";
        System.out.println(cyan("Question: ") + question);

        Response<AiMessage> response = chatModel.generate(UserMessage.from( question));
        prettyPrint(response);
    }

    @Test
    public void should_tune_llm_request() {
        System.out.println(yellow("Tuning Chat Model:"));

        // Use OpenAI's GPT model for generating chat completions
        ChatLanguageModel chatModel = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))

                // TUNING MODEL
                .temperature(0.7d)         // Same temperature setting
                .topP(0.8d)                // OpenAI equivalent for top_p (nucleus sampling)
                .maxTokens(2000)           // Max tokens (same as max_output_tokens)
                .maxRetries(3)             // Retry logic (OpenAI equivalent)
                // <---

                .build();

        String question = "What are the profiles of Devoxx attendees";
        Response<AiMessage> response = chatModel.generate(UserMessage.from(question));

        prettyPrint(response);
    }

    /** Streaming handler to log results and release the latch on completion. */
    @RequiredArgsConstructor
    public static class PrettyPrintStreamingResponseHandler implements StreamingResponseHandler<AiMessage> {

        private final CountDownLatch latch;

        @Override
        public void onNext(String s) { System.out.print(s); }

        @Override
        public void onComplete(Response<AiMessage> response) {
            latch.countDown();
        }

        @Override
        public void onError(Throwable throwable) {
            System.err.println("Error : " + throwable.getMessage());
            latch.countDown();
        }
    }

    @Test
    public void should_chat_language_model_streaming() {
        System.out.println(yellow("Using Chat Model (Streaming):"));
        String question = "Give me a poem in 20 sentences about DEVOXX";
        System.out.println(cyan("Question: ") + question);

        CountDownLatch latch = new CountDownLatch(1);

        getChatLanguageModelStreaming().generate(question, new PrettyPrintStreamingResponseHandler(latch));

        try {
            if (!latch.await(15, TimeUnit.SECONDS)) {
                System.out.println("Streaming did not finish in time.");
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}
