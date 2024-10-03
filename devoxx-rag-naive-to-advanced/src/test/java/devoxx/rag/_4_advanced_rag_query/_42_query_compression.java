package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

public class _42_query_compression extends AbstractDevoxxTest {


    @Test
    public void shouldTestQueryCompression() {

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                // Commons Retriever
                .contentRetriever(createRetriever("/text/johnny.txt"))
                // Add a Query Transformation
                .queryTransformer(new CompressingQueryTransformer(getChatLanguageModel(MODEL_GEMINI_PRO)))
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(getChatLanguageModel(MODEL_GEMINI_PRO))
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        System.out.println(assistant.answer("Give me the name of the horse"));
        System.out.println(assistant.answer("Can you tell where he lives ?"));
        System.out.println(assistant.answer("What does he do ?"));

    }

}