#  Classificador de imagens - thumb up or down

Para rodar esta aplicação, instale o pacote Anaconda, disponível em https://www.anaconda.com/products/individual
Os arquivos estão em .py, sendo recomendada a aplicação Spyder para rodá-los. 

Por restrições de tempo, o classificador não foi integrado a um front-end, sendo possível classificar imagens conforme indicado nos comentários do arquivo.
Além disso, o dataset foi improvisado para a execução do exercício, dentro da viabilidade de tempo e recursos.

Classificador de imagens:

- classificador.py

Outros arquivos:

- convolutional_neural_network
- model50epoc - modelo com maior número de acertos. Aparenta estar com probelmas de overfitting, mas foi o melhor que consegui dentro do tempo disponível;
- perf50epoc - gráfico com métricas do modelo atual
- model30epoc - modelo alternativo, com early stop para reduzir overffiting - menos acertos
- perf30epoc - gráfico com métricas do modelo alternativo

Dataset:

- 163 imagens de duas classes("thumbs up" e "thumbs down") para treino, consistindo em imagens de internet e fotos de celular.
- 39 imagens de duas classes("thumbs up" e "thumbs down") para teste, consistindo em imagens de internet e fotos de celular.
- 20 imagens de duas classes("thumbs up" e "thumbs down") para demosntração, consistindo em fotos de celular.




