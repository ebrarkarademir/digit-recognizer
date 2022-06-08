import pygame
import pandas as pd
import keras
import numpy as np
from PIL import Image
from numpy import asarray


WIDTH, HEIGHT = 218, 218
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

MODEL = keras.models.load_model("models/digit_predicter_cnn")

EVENT = pygame.USEREVENT + 1

FPS = 120
pygame.display.set_caption("draw_digit_")

def capture():
    pygame.image.save(WIN,"imgs/digit_img.jpg")
    print('Screen saved!')

def predict(key_pressed):
    capture()
    if  key_pressed[pygame.K_SPACE]:
        print("Making prediction on drawing...")
        image = Image.open("imgs/digit_img.jpg")
        image = image.resize((28, 28))
        image = image.convert('L')
        image = asarray(image).reshape(28, 28, 1) / 255
        image = np.array([image])
        prediction = MODEL.predict(image).argmax()
        probability = MODEL.predict(image)[0][prediction]
        print(f"Prediction is {prediction} with %{round(probability*100,5)} probability")


def draw_window(pos, poss):
    WIN.fill(BLACK)
    if pos != (0,0):
        for i in poss:
            pygame.draw.circle(WIN, WHITE, i, 5, 5)

    pygame.display.update()

def main():
    clock = pygame.time.Clock()
    run = True
    drawing = False
    pos = (0,0)
    poss = []

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == EVENT:
                pass

            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False

            if event.type == pygame.MOUSEMOTION:
                if drawing:
                    pos = pygame.mouse.get_pos()
                    poss.append(pos)

            if event.type == pygame.KEYDOWN:
                key_pressed = pygame.key.get_pressed()
                if key_pressed[pygame.K_LCTRL]:
                    poss.clear()
                else:
                    predict(key_pressed)




        draw_window(pos, poss)

    pygame.quit()


if __name__ == "__main__":
    main()