package devoxx.rag;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.File;
import java.util.concurrent.CopyOnWriteArrayList;

public class ExtendedInMemoryEmbeddingStore extends InMemoryEmbeddingStore<TextSegment> {

    public static ExtendedInMemoryEmbeddingStore init() {
        return new ExtendedInMemoryEmbeddingStore();
    }

    public static ExtendedInMemoryEmbeddingStore init(ExtendedInMemoryEmbeddingStore fileStore) {
        ExtendedInMemoryEmbeddingStore store = new ExtendedInMemoryEmbeddingStore();
        store.getEntries().addAll(fileStore.getEntries());
        return store;
    }

    public static ExtendedInMemoryEmbeddingStore init(@Nullable File file) {
        var store = new ExtendedInMemoryEmbeddingStore();
        if (file != null) {
            ensureFileExists(file);
            var fileStore = InMemoryEmbeddingStore.fromFile(file.toPath());
            if (fileStore != null) {
                store.getEntries().addAll(obtainEntries(fileStore));
            }
        }
        return store;
    }

    public ExtendedInMemoryEmbeddingStore() {
        super();
    }

    public final CopyOnWriteArrayList<TextSegment> getEntries() {
        return obtainEntries(this);
    }

    private static CopyOnWriteArrayList<TextSegment> obtainEntries(InMemoryEmbeddingStore<TextSegment> _this) {
        try {
            var entriesField = InMemoryEmbeddingStore.class.getDeclaredField("entries");
            entriesField.setAccessible(true);
            //noinspection unchecked
            return (CopyOnWriteArrayList<TextSegment>) entriesField.get(_this);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Failed to access entries field via reflection", e);
        }
    }

    private static void ensureFileExists(@NotNull File file) {
        if (!file.exists()) {
            try {
                if (!file.createNewFile()) {
                    throw new RuntimeException("Failed to create file: " + file);
                }
            } catch (Exception e) {
                throw new RuntimeException("Failed to create file: " + file, e);
            }
        }
    }
}