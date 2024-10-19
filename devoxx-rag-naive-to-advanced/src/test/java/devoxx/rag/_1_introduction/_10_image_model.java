package devoxx.rag._1_introduction;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.openai.OpenAiImageModel;
import dev.langchain4j.model.output.Response;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

public class _10_image_model {

    @Test
    public void shouldSayHelloToImageModel() {
        // Use OpenAI's DALL·E model for image generation
        ImageModel imageModel = OpenAiImageModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .build();

        Response<Image> imageListResponse = imageModel
                .generate("Photo of a sunset over Malibu beach");

        var img = imageListResponse.content();
        assertThat(img.url()).isNotNull();
        assertThat(img.base64Data()).isNotNull();
        System.out.println(img.url()); // Prints out the URL for each generated image
    }}
