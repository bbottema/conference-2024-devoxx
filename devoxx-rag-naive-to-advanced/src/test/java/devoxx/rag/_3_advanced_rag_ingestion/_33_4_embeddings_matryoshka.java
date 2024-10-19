package devoxx.rag._3_advanced_rag_ingestion;

import com.datastax.astra.client.exception.TooManyDocumentsToCountException;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.vertexai.VertexAiEmbeddingModel;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

public class _33_4_embeddings_matryoshka extends AbstractDevoxxTest {

    @Test
    public void should_compare_dimensionality() throws TooManyDocumentsToCountException {
        var embeddingModel768 = getEmbeddingModelBuilder(MODEL_OPENAI_TEXT3_SMALL).dimensions(768).build();
        var embeddingModel256 = getEmbeddingModelBuilder(MODEL_OPENAI_TEXT3_SMALL).dimensions(256).build();

        Response<Embedding> embeddingOne768 = embeddingModel768.embed("young dog");
        Response<Embedding> embeddingTwo256 = embeddingModel256.embed("young dog");

        System.out.println("embeddingOne768 = " + embeddingOne768.content().vectorAsList());
        System.out.println("embeddingTwo256 = " + embeddingTwo256.content().vectorAsList());

        System.out.println(embeddingTwo256.content().vectorAsList().equals(
            embeddingOne768.content().vectorAsList().subList(0, 256)));
    }
}
