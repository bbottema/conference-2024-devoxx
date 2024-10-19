package devoxx.rag._1_introduction;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingStore;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.io.File;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _13_embedding_store extends AbstractDevoxxTest {

    static final File INTRO_STORE = new File("src/test/resources/intro_store.json");

    @Test
    public void should_connect_store() {
        System.out.println(yellow("Connect Vector Database"));

        // Create Collection
        ExtendedInMemoryEmbeddingStore vectorStore = ExtendedInMemoryEmbeddingStore.init(INTRO_STORE);
        System.out.println(cyan("[OK] ") + " Collection Created");

        // Insert a document
        vectorStore.removeAll();
        vectorStore.add(
                new Embedding(new float[]{0.2f, 0.2f, 0.2f, 0.2f, 0.2f}),
                new TextSegment("Hello World", new Metadata())
        );
        System.out.println(cyan("[OK] ") + " Document inserted");

        // With LangChain4J
        EmbeddingStore<TextSegment> embeddingStore = ExtendedInMemoryEmbeddingStore.init(vectorStore);
        embeddingStore.add(Embedding.from(new float[] {.2f, .2f, .2f, .2f, .2f}), TextSegment.from("Hello World"));
        System.out.println(cyan("[OK] ") + " Document inserted with store");

        vectorStore.serializeToFile(INTRO_STORE.toPath());
    }

    @Test
    public void deleteCollection() {
        System.out.println(yellow("Delete Collection"));
        ExtendedInMemoryEmbeddingStore vectorStore = ExtendedInMemoryEmbeddingStore.init(INTRO_STORE);
        vectorStore.removeAll();
        vectorStore.serializeToFile(INTRO_STORE.toPath());
        System.out.println(cyan("[OK] ") + " Collection Deleted");
    }
}
