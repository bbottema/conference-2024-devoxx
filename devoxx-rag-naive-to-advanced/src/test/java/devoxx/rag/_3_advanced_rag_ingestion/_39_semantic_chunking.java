package devoxx.rag._3_advanced_rag_ingestion;

import dev.langchain4j.data.document.splitter.DocumentBySentenceSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.store.embedding.CosineSimilarity;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.experiments.Utils;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.IntStream;

import static com.datastax.astra.internal.utils.AnsiUtils.magenta;
import static com.datastax.astra.internal.utils.AnsiUtils.yellow;
import static java.util.Map.Entry.comparingByKey;

public class _39_semantic_chunking extends AbstractDevoxxTest {
    @Test
    public void semanticChunking() {
        String text = loadDocumentText("text/berlin.txt").text();

        var embeddingModel = getEmbeddingModel();

        // split by sentences
        DocumentBySentenceSplitter splitter = new DocumentBySentenceSplitter(200, 20);
        List<String> sentences = Arrays.asList(splitter.split(text));

        // create groups of sentences (1 before, 2 after current sentence)
        List<List<String>> slidingWindowSentences = Utils.slidingWindow(sentences, 1, 2);
        List<TextSegment> concatenatedSentences = slidingWindowSentences.stream()
                .map(strings -> TextSegment.from(String.join(" ", strings)))
                .toList();

        // calculate vector embeddings for each of these sentence groups
        List<Embedding> embeddings = embeddingModel.embedAll(concatenatedSentences).content();

        // calculate the pair-wise similarities between each sentence group in parallel
        List<Double> similarities = IntStream.range(0, embeddings.size() - 1)
                .parallel()
                .mapToDouble(i -> CosineSimilarity.between(embeddings.get(i), embeddings.get(i + 1)))
                .boxed()
                .toList();

        // pair each similarity with its index
        List<Map.Entry<Double, Integer>> similaritiesWithIndices = IntStream.range(0, similarities.size())
                .parallel()
                .mapToObj(i -> Map.entry(similarities.get(i), i))
                .sorted(comparingByKey()) // Sort the similarities to find the lowest ones
                .toList();

        // Extract the indices of the 100 lowest similarities
        List<Integer> lowestSimilaritiesIndices = similaritiesWithIndices.stream()
                .limit(100)
                .map(Map.Entry::getValue)
                .sorted()
                .toList();

        System.out.println(magenta("Lowest similarity breakpoints = ") + lowestSimilaritiesIndices);

        List<String> finalSentenceGroups = new ArrayList<>();

        int startIndex = 0;
        for (int lowestSimilaritiesIndex : lowestSimilaritiesIndices) {
            finalSentenceGroups.add(String.join(" ", sentences.subList(startIndex, lowestSimilaritiesIndex)));
            startIndex = lowestSimilaritiesIndex;
        }
        finalSentenceGroups.add(String.join(" ", sentences.subList(startIndex, sentences.size())));

        ScoringModel scoringModel = getScoringModel();

        // Process final sentence groups in parallel
        List<String> results = finalSentenceGroups.parallelStream()
                .map(sentenceGroup -> {
                    Double score = scoringModel.score(sentenceGroup, "What is the population of Berlin?").content();
                    return score > 0.7
                            ? String.format(yellow("—".repeat(10) + " Ranking score: %s " + "—".repeat(60)) + "%n%s", score, sentenceGroup)
                            : null;
                })
                .filter(Objects::nonNull)
                .toList();

        // Print the main results sequentially
        System.out.println(magenta("\nResulting entries: " + results.size()));
        System.out.println(magenta("\nHighest ranking:\n" + results.getFirst()));
        System.out.println(magenta("\nLowest ranking:\n" + results.getLast()));

    }
}
