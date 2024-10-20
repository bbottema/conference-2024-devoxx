package devoxx.rag._4_advanced_rag_query;

import dev.langchain4j.classification.EmbeddingModelTextClassifier;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import org.junit.jupiter.api.Test;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static com.datastax.astra.internal.utils.AnsiUtils.yellow;

public class _41_4_query_routing_with_classifier extends AbstractDevoxxTest {

    @Test
    public void testQueryRouting() {
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(new LlmRouter())
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .retrievalAugmentor(retrievalAugmentor)
                .chatLanguageModel(getChatLanguageModel())
                .build();

        System.out.println(assistant.answer("Give me the name of the horse"));
        System.out.println(assistant.answer("Give me the name of the dog"));
    }

    /**
     * Custom Router
     */
    private static class LlmRouter extends AbstractDevoxxTest implements QueryRouter  {

        enum Category {
            DOG, HORSE
        }

        @Override
        public Collection<ContentRetriever> route(Query query) {
            EmbeddingModel embeddingModel = getEmbeddingModel();

            var classifier =
                new EmbeddingModelTextClassifier<>(embeddingModel, Map.of(
                    Category.DOG, List.of("something about dogs", "dog, dogs, and puppies", "dog species"),
                    Category.HORSE, List.of("something about horses", "horse racing", "what kind of horse is it?")
                ));

            List<Category> category = classifier.classify(query.text());

            System.out.println(yellow("-> Category recognized: " + category));

            if (Category.HORSE.equals(category.getFirst())) {
                return List.of(createRetriever("/text/johnny.txt"));
            } else if (Category.DOG.equals(category.getFirst())) {
                return  List.of(createRetriever("/text/shadow.txt"));
            }

            return Collections.emptyList();
        }
    }
}