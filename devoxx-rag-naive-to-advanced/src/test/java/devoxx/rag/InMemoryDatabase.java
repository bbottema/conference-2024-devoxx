package devoxx.rag;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Mimics an in-memory database for documents. */
public class InMemoryDatabase {

    // Store collections, each collection is a Map of documents
    private final Map<String, Collection> collections = new ConcurrentHashMap<>();

    // Create a collection
    public Collection createCollection(String name, int dimension) {
        Collection collection = new Collection(name);
        collections.put(name, collection);
        return collection;
    }

    // Get a collection by name
    public Collection getCollection(String name) {
        return collections.get(name);
    }

    // Inner class to represent a collection
    public static class Collection {
        private final String name;
        private final Map<String, Map<String, Object>> documents = new ConcurrentHashMap<>();

        public Collection(String name) {
            this.name = name;
        }

        // Insert a document into the collection
        public void insertDocument(String documentId, Map<String, Object> document) {
            documents.put(documentId, document);
        }

        // Retrieve a document by ID
        public Map<String, Object> getDocument(String documentId) {
            return documents.get(documentId);
        }

        // Get all documents
        public Map<String, Map<String, Object>> getAllDocuments() {
            return documents;
        }

        // For cosine similarity mockup, this would typically calculate similarity between vectors
        public Map<String, Object> findSimilarDocument(Map<String, Object> vector) {
            // Just a stub, real cosine similarity would be more complex
            return documents.values().iterator().next(); // Mock: return the first document
        }
    }
}
