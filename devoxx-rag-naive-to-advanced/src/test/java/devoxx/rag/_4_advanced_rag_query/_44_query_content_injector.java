package devoxx.rag._4_advanced_rag_query;

import com.datastax.astra.langchain4j.store.embedding.AstraDbEmbeddingStore;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.injector.ContentInjector;
import dev.langchain4j.rag.content.injector.DefaultContentInjector;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import devoxx.rag.AbstractDevoxxTest;
import devoxx.rag.Assistant;
import devoxx.rag.ExtendedInMemoryEmbeddingStore;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;

import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;


public class _44_query_content_injector extends AbstractDevoxxTest {

    static final File QUOTE_PREPOPULATED_STORE = new File("src/test/resources/quote_prepopulated_store.json");

    @Test
    void should_inject_single_content() {
        UserMessage userMessage = UserMessage.from("Tell me about bananas.");
        List<Content> contents = singletonList(Content.from("Bananas are awesome!"));
        ContentInjector injector = new DefaultContentInjector();
        UserMessage injected = injector.inject(contents, userMessage);
        System.out.println(injected.text());
    }

    @Test
    public void shouldRetrieveDocument() {
        // Retrieving the content from the embedding store
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(ExtendedInMemoryEmbeddingStore.init(QUOTE_PREPOPULATED_STORE))
                .embeddingModel(getEmbeddingModel())
                .maxResults(2)
                .minScore(0.5)
                .build();

        // Enhance the content retriever to add meta data in the prompt
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                // Query injector
                .contentInjector(DefaultContentInjector.builder()
                        // Prompt Template
                        .promptTemplate(PromptTemplate.from("The document is about %s. The MD5 hash is %s."))
                        // Values to inject
                        .metadataKeysToInclude(asList("document_format", "md5"))
                        .build())
                .build();

        // configuring it to use the components we've created above.
        Assistant ai = AiServices.builder(Assistant.class)
                .retrievalAugmentor(retrievalAugmentor)
                .chatLanguageModel(getChatLanguageModel())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        String response = ai.answer("What did Aristotle say about the good life?");
        System.out.println(response);
    }

}