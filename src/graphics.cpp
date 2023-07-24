#include "../header/graphics.hpp"

void Graphics::show(const TensorInline& t, const int label) {
    
    const int width = t.getWidth();
    const int height = t.getHeight();

    sf::RenderWindow window(sf::VideoMode(width, height), "Matri s");

    sf::Image image;
    image.create(width, height);

    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            double value = t.tensor[y * width + x];
            // Échelonner la valeur dans l'intervalle 0-255 pour l'utilisation des couleurs RVB
            int colorValue = static_cast<int>((value + 1) * 0.5 * 255);
            sf::Color color(colorValue, colorValue, colorValue);
            image.setPixel(x, y, color);
        }
    }

    sf::Texture texture;
    texture.loadFromImage(image);

    sf::Sprite sprite(texture);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }
}