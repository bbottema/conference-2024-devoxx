package devoxx.rag;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import dev.langchain4j.store.embedding.CosineSimilarity;
import lombok.RequiredArgsConstructor;

import java.util.List;
import java.util.ArrayList;

import static devoxx.rag.AbstractDevoxxTest.getEmbeddingModel;

@RequiredArgsConstructor
public class BasicEmbeddingModelBasedScoringModel implements ScoringModel {

    private final EmbeddingModel embeddingModel;

    @Override
    public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
        // Embed the query
        Response<Embedding> queryEmbeddingResponse = embeddingModel.embed(query);
        Embedding queryEmbedding = queryEmbeddingResponse.content();

        // Prepare the list of scores
        List<Double> scores = new ArrayList<>();

        // For each segment, calculate the cosine similarity score
        for (TextSegment segment : segments) {
            // Embed the segment text
            Response<Embedding> segmentEmbeddingResponse = embeddingModel.embed(segment.text());
            Embedding segmentEmbedding = segmentEmbeddingResponse.content();

            // Compute the cosine similarity score
            double score = CosineSimilarity.between(segmentEmbedding, queryEmbedding);
            scores.add(score);
        }

        // Return the scores with token usage (using token usage from the last embedding response)
        return Response.from(scores, queryEmbeddingResponse.tokenUsage(), queryEmbeddingResponse.finishReason());
    }
}
