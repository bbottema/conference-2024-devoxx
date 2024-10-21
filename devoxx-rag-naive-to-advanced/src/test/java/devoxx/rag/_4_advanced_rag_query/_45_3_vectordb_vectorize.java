package devoxx.rag._4_advanced_rag_query;

import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Disabled;

@Disabled("This test depends on .addAllVectorize and makes this test AstraDb-centric")
public class _45_3_vectordb_vectorize extends AbstractDevoxxTest {

//    private static final String COLLECTION_NAME = "quote";
//
//    @Test
//    public void should_vectorize() throws URISyntaxException {
//        // Setup the store
//        CollectionOptions collectionOptions = CollectionOptions
//                .builder()
//                .vectorSimilarity(SimilarityMetric.COSINE)
//                .vectorize("nvidia","NV-Embed-QA")
//                .build();
//        Collection<Document> collection = new DataAPIClient(ASTRA_TOKEN,
//                DataAPIOptions.builder().withObserver(new LoggingCommandObserver(AbstractDevoxxTest.class)).build())
//                .getDatabase("https://3a2670a5-adbb-449e-b744-16d5182f5b70-us-east-2.apps.astra.datastax.com")
//                .createCollection("vectorize_test", collectionOptions);
//        AstraDbEmbeddingStore embeddingStore = new AstraDbEmbeddingStore(collection);
//
//        // Ingest documents
//        DocumentSplitter splitter = DocumentSplitters.recursive(300, 20);
//
//        // Add and compute vectors on the SPOT
//        embeddingStore.getCollection().deleteAll();
//        embeddingStore.addAllVectorize(splitter.split(loadDocumentText("text/johnny.txt")));
//        embeddingStore.addAllVectorize(splitter.split(loadDocumentText("text/shadow.txt")));
//        embeddingStore.addAllVectorize(splitter.split(loadDocumentText("text/berlin.txt")));
//
//        // Search
//        EmbeddingSearchRequestAstra searchQuery = EmbeddingSearchRequestAstra.builderAstra()
//                .maxResults(10)
//                .minScore(0.1d)
//                .queryVectorize("What is the Name of the HORSE ?")
//                .build();
//        embeddingStore.search(searchQuery).matches()
//                .stream()
//                .map(match -> match.embedded().text())
//                .forEach(System.out::println);
//    }
}
