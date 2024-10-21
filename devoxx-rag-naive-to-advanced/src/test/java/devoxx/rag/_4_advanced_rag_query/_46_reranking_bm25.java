package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import devoxx.rag.rerank.bm25.Bm25ScoringModel;
import devoxx.rag.rerank.bm25.Language;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

import static devoxx.rag._3_advanced_rag_ingestion._39_custom_ingestion.QUOTE_PREPOPULATED_STORE;

public class _46_reranking_bm25 extends AbstractDevoxxTest {

    private static final ExtendedInMemoryEmbeddingStore DATABASE = ExtendedInMemoryEmbeddingStore.init(QUOTE_PREPOPULATED_STORE);

    @Test
    public void should_search_in_vector_db() throws IOException {
        // I have to create a EmbeddingModel
        EmbeddingModel embeddingModel = getEmbeddingModel();
        // Embed the question
        String question = "We struggle all our life for nothing";
        Embedding questionEmbedding = embeddingModel.embed(question).content();
        // We need the store
        // Build the Search Query
        EmbeddingSearchRequest searchQuery = EmbeddingSearchRequest.builder()
                .queryEmbedding(questionEmbedding)
                .maxResults(25)   // increase the number of users
                .minScore(0.1d)  // similarity score low to get more results
                .build();

        // Execute the request
        List<EmbeddingMatch<TextSegment>> matches = DATABASE.search(searchQuery).matches();
        matches.forEach(match -> {
            BigDecimal bigDecimal = BigDecimal.valueOf(match.score()).setScale(4, RoundingMode.HALF_UP);
            System.out.printf("Similarity: %s - %s%n", bigDecimal, match.embedded().text());
        });

        // ReRanking
        List<TextSegment> chunks = matches.stream().map(EmbeddingMatch::embedded).toList();
        List<Double> scores = new Bm25ScoringModel(Language.ENGLISH).scoreAll(chunks, question).content();
        for (int i = 0; i < chunks.size(); i++) {
            System.out.printf("BM25 Score: %s - %s%n", scores.get(i), chunks.get(i).text());
        }
    }
}
