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
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Map;

import static dev.langchain4j.store.embedding.filter.MetadataFilterBuilder.metadataKey;

@Slf4j
public class _39_custom_ingestion extends AbstractDevoxxTest {

    public static final File QUOTE_PREPOPULATED_STORE = new File("src/test/resources/quote_prepopulated_store.json");

    private static final ExtendedInMemoryEmbeddingStore DATABASE = ExtendedInMemoryEmbeddingStore.init(QUOTE_PREPOPULATED_STORE);

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
        TextSegment textSegment = TextSegment.from(quote.body(), new Metadata(Map.of(
                "authors", quote.author(),
                "tags", quote.tags().toString(),
                "md5", computeMD5(quote.body()),
                "document_format", "dummy_format" // used in later tests, but we don't care about it here (it's something Astra's database does)
        )));
        Embedding embedding = embeddingModel.embed(textSegment).content();
        return new Document(embedding, textSegment);
    }

    // Utility function to compute MD5 hash
    private static String computeMD5(String content) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hash = md.digest(content.getBytes(StandardCharsets.UTF_8));
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                hexString.append(String.format("%02x", b));
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("Error generating MD5 hash", e);
        }
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
