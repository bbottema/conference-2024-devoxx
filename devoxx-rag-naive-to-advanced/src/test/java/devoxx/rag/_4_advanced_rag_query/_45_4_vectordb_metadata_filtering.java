package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.client.exception.TooManyDocumentsToCountException;
import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.client.model.Filters.eq;
import static com.datastax.astra.client.model.Filters.lt;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;


public class _45_4_vectordb_metadata_filtering extends AbstractDevoxxTest {

    static final String COLLECTION_NAME = "quote";

    @Test
    public void should_filter_on_metadata() throws TooManyDocumentsToCountException {
        System.out.println(yellow("Count documents"));
        System.out.println(getCollection(COLLECTION_NAME).countDocuments(1000));

        // List me all quotes from Aristotle and show me the quote and tags
        System.out.println(yellow("Show Aristotle quotes"));

        getCollection(COLLECTION_NAME)
                .find(eq("authors", "aristotle"))
                .forEach(doc -> {System.out.println(doc.get("content")); });
    }

    @Test
    public void shouldRetrieveDocument() {
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(new AstraDbEmbeddingStore(getCollection(COLLECTION_NAME)))
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