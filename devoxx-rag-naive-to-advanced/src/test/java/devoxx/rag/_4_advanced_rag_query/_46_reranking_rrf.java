package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.internal.utils.AnsiUtils;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.filter.comparison.IsEqualTo;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import devoxx.rag._3_advanced_rag_ingestion._39_custom_ingestion;
import devoxx.rag.rerank.rrf.ReciprocalRankFusion;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class _46_reranking_rrf extends AbstractDevoxxTest  {

    private static final ExtendedInMemoryEmbeddingStore DATABASE = ExtendedInMemoryEmbeddingStore.init(_39_custom_ingestion.QUOTE_PREPOPULATED_STORE);

    @Test
    public void testRerankingRRF() {
        String question = "We struggle all our life for nothing";
        Embedding questionEmbedding = getEmbeddingModel().embed(question).content();
        ExtendedInMemoryEmbeddingStore embeddingStore = DATABASE;

        EmbeddingSearchResult<TextSegment> aristotleResults = embeddingStore.search(EmbeddingSearchRequest.builder()
                .maxResults(10)
                .minScore(0.1)
                .queryEmbedding(questionEmbedding)
                .filter(new IsEqualTo("authors", "aristotle"))
                .build());

        EmbeddingSearchResult<TextSegment> plateResults = embeddingStore.search(EmbeddingSearchRequest.builder()
                .maxResults(10)
                .minScore(0.1)
                .queryEmbedding(questionEmbedding)
                .filter(new IsEqualTo("authors", "plato"))
                .build());

        System.out.println(AnsiUtils.yellow("=========== ARISTOTLE ============"));
        aristotleResults.matches().forEach(match -> {
            System.out.println(AnsiUtils.cyan(BigDecimal.valueOf(match.score()).setScale(4, RoundingMode.HALF_UP).toString()) + " - " + match.embedded().text());
        });
        System.out.println(AnsiUtils.yellow("============= PLATO =============="));
        plateResults.matches().forEach(match -> {
            System.out.println(AnsiUtils.cyan(BigDecimal.valueOf(match.score()).setScale(4, RoundingMode.HALF_UP).toString()) + " - " + match.embedded().text());
        });

        // RRF
        List<TextSegment> aristotleList = aristotleResults.matches().stream().map(EmbeddingMatch::embedded).toList();
        List<TextSegment> plateList     = plateResults.matches().stream().map(EmbeddingMatch::embedded).toList();
        Map<TextSegment, Double> fusedResults = new ReciprocalRankFusion().score(Arrays.asList(aristotleList, plateList));
        System.out.println(AnsiUtils.yellow("============= RRF =============="));
        fusedResults.entrySet()
                .stream()
                .sorted((e1, e2) -> Double.compare(e2.getValue(), e1.getValue()))
                .forEach(entry -> {
                    BigDecimal score = BigDecimal.valueOf(entry.getValue()).setScale(4, RoundingMode.HALF_UP);
                    System.out.printf("%s - %s%n", AnsiUtils.cyan(score.toString()), entry.getKey().text());
                });

    }
}
