package devoxx.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.lang.reflect.Field;
import java.util.concurrent.CopyOnWriteArrayList;

public class ExtendedInMemoryEmbeddingStore<T> extends InMemoryEmbeddingStore<T> {

    public ExtendedInMemoryEmbeddingStore() {
        super();
    }

    public final CopyOnWriteArrayList<?> getEntries() {
        try {
            Field entriesField = InMemoryEmbeddingStore.class.getDeclaredField("entries");
            entriesField.setAccessible(true);  // Bypass private access
            return (CopyOnWriteArrayList<?>) entriesField.get(this);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Failed to access entries field via reflection", e);
        }
    }
}