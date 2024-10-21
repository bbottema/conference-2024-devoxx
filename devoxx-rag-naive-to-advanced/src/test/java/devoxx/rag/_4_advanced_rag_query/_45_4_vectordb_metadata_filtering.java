package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.client.exception.TooManyDocumentsToCountException;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static devoxx.rag._3_advanced_rag_ingestion._39_custom_ingestion.QUOTE_PREPOPULATED_STORE;

// almost identical to _39_custom_ingestion.java#langchain4jEmbeddingStore()
public class _45_4_vectordb_metadata_filtering extends AbstractDevoxxTest {

    private static final ExtendedInMemoryEmbeddingStore DATABASE = ExtendedInMemoryEmbeddingStore.init(QUOTE_PREPOPULATED_STORE);

    @Disabled("This test depends on AstraDb-specific filtering, eq()")
    @Test
    public void should_filter_on_metadata() throws TooManyDocumentsToCountException {
//        System.out.println(yellow("Count documents"));
//        System.out.println(getCollection(COLLECTION_NAME).countDocuments(1000));
//
//        // List me all quotes from Aristotle and show me the quote and tags
//        System.out.println(yellow("Show Aristotle quotes"));
//
//        getCollection(COLLECTION_NAME)
//                .find(eq("authors", "aristotle"))
//                .forEach(doc -> {System.out.println(doc.get("content")); });
    }

    @Test
    public void shouldRetrieveDocument() {
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(DATABASE)
                .embeddingModel(getEmbeddingModel())
                .filter(new IsEqualTo("authors", "aristotle"))
                .maxResults(2)
                .minScore(0.5)
                .build();

        Assistant ai = AiServices.builder(Assistant.class)
                .contentRetriever(contentRetriever)
                .chatLanguageModel(getChatLanguageModel())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        String response = ai.answer("What did Aristotle say about the good life?");
        System.out.println(response);
    }
}