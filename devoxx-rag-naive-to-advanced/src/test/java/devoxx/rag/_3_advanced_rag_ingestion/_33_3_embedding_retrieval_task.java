package devoxx.rag._3_advanced_rag_ingestion;

import com.datastax.astra.client.exception.TooManyDocumentsToCountException;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.CosineSimilarity;
import devoxx.rag.AbstractDevoxxTest;
import org.junit.jupiter.api.Test;

public class _33_3_embedding_retrieval_task extends AbstractDevoxxTest {

    @Test
    public void should_compare_similarity() throws TooManyDocumentsToCountException {
        var embeddingModel = getEmbeddingModel();

        var definitionText = """
            Embedding models are machine learning models that convert complex data, like text or images, 
            into numerical representations called embeddings. These embeddings capture the relationships 
            between different pieces of data, allowing machines to understand and process them more 
            effectively. They are used in various applications, including natural language processing, 
            image and video analysis, and recommendation systems.""";

        Response<Embedding> definitionEmbedding = embeddingModel.embed(definitionText);

        var questionText = "What are embedding models?";

        Response<Embedding> questionEmbedding = embeddingModel.embed(questionText);

        double similarity = CosineSimilarity.between(
            definitionEmbedding.content(),
            questionEmbedding.content()
        );

        System.out.println("similarity = " + similarity);
    }
}
