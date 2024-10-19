package devoxx.rag._1_introduction;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

import static com.datastax.astra.internal.utils.AnsiUtils.cyan;

public class _12_embedding_model extends AbstractDevoxxTest {

    @Test
    public void should_illustrate_embedding_model() {
        String chunk = "HELLO this is a vector";

        // Use OpenAI's embedding model
        EmbeddingModel embeddingModel = OpenAiEmbeddingModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("text-embedding-ada-002") // This is the common embedding model from OpenAI
                .build();

        Response<Embedding> res = embeddingModel.embed(chunk);

        // The Model has a dimensionality of 1536
        System.out.println(cyan("Vector: ") + res.content().vectorAsList());
        System.out.println(cyan("Dimensionality: ") + res.content().dimension());
    }
}
