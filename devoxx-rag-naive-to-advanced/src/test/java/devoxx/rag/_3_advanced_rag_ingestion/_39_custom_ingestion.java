package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import devoxx.rag.ExtendedInMemoryEmbeddingStore.Document;
import devoxx.rag.Quote;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;

@Slf4j
class _39_custom_ingestion extends AbstractDevoxxTest {

    static final File QUOTE_PREPOPULATED_STORE = new File("src/test/resources/quote_prepopulated_store.json");

    public static ExtendedInMemoryEmbeddingStore getQuotePrepopulatedEmbeddingStore() {
        return ExtendedInMemoryEmbeddingStore.init(QUOTE_PREPOPULATED_STORE);
    }

    private static final ExtendedInMemoryEmbeddingStore DATABASE = getQuotePrepopulatedEmbeddingStore();

    @Test
    void shouldIngestDocuments() throws IOException {
        EmbeddingModel embeddingModel = getEmbeddingModel();
        DATABASE.removeAll();
        loadQuotes("/json/philo_quotes.json")       // extraction
                .parallelStream()
                .peek(quote -> System.out.println("quote = " + quote))
                .map(quote -> embedAsDocument(quote, embeddingModel))
                .forEach((Document document) -> DATABASE.add(document.getEmbedding(), document.getTextSegment()));

        DATABASE.serializeToFile(QUOTE_PREPOPULATED_STORE.toPath());
    }

    @NotNull
    private static Document embedAsDocument(Quote quote, EmbeddingModel embeddingModel) {
        Embedding embedding = embeddingModel.embed(quote.body()).content();
        TextSegment textSegment = TextSegment.from(quote.body(), new Metadata(Map.of(
                "authors", quote.author(),
                "tags", quote.tags().toString()
        )));
        return new Document(embedding, textSegment);
    }

    @Test
    void langchain4jEmbeddingStore() {
        // I have to create a EmbeddingModel
        EmbeddingModel embeddingModel = getEmbeddingModel();

        // Embed the question
        Embedding questionEmbedding = embeddingModel.embed("We struggle all our life for nothing").content();

        // Query with a filter(2)
        log.info("Querying with filter");
        DATABASE.search(EmbeddingSearchRequest.builder()
                        .queryEmbedding(questionEmbedding)
                        .filter(metadataKey("authors").isEqualTo("aristotle"))
                        .maxResults(3).minScore(0.1d).build())
                .matches()
                .stream().map(embeddingMatch -> embeddingMatch.embedded().text())
                .forEach(System.out::println);
    }

}
