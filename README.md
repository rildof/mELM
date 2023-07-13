# Rede Neural Extrema (mELM)
Esse repositório tem a intenção de estudar e desenvolver um antivírus auto-explicável com o uso da Rede Neural Extrema(mELM), um tipo de rede neural que tem a vantagem de ter uma grande eficiência computacional, boa generalização a partir de um número pequeno de neurônios na camada oculta. Isso será utilizado a nosso favor, simplificando ainda mais a camada escondida.

## Simplificação da camada escondida
A ideia da simplificação da rede neural escondida está em observar a média, ou a moda da base de dados de treinamento, e utilizar isso como os pesos da camada escondida. Realizar a validação do treino com esses pesos e retirar os valores que foram treinados corretamente, para que a próxima iteração realize o treinamento apenas com os casos mal-sucedidos da iteração anterior.\\
Isso diminuiria em grande escala o número de neurônios na camada oculta e traria ainda mais eficiência no sistema. O processo de pensamento para implementação pode ser observado abaixo:\\

>1- Realizar o cálculo com toda a base de dados inicialmente, usando apenas dois neurônios para o caso binário, criando assim uma matriz
de pesos 2xN, sendo N o número de entradas.

>2- Realizar o treino, e guardar esses neurônios criados. Após isso, remover da base de dados as classes e contra-classes que esses pesos satisfizeram.

>3- Repetir o treino, agora apenas com os dados que deram errado na iteração anterior. Repetir o processo até que a acurácia seja ao menos próxima de 100%.

>4- Após isso, realizar o treino com 2*K camadas escondidas, sendo K o número de iterações.


## Arquivos .py
O arquivo ´melm.py´ é o arquivo inicial do inicio da Iniciação Científica. Ele foi o ponto de partida do estudo.\\
O arquivo ´melm_R2.py´ contém a implementação do sistema de simplificação da camada escondida, e executa, assim como o melm.py, a base de dados da diabetes, que é uma base de dados para teste.\\
O arquivo ´melm_R3.py´ contém a adição anterior, e também a implementação de uma função que permite adicionar bases de dados externas e dentro do próprio programa criar o Dataset de treino e de teste. Todas modificações posteriores serão feitas exclusivamente no arquivo ´melm_R3.py´.


## Executar exemplos
Uma base de dados de teste para diabetes pode ser executada a partir dos seguintes comandos:
para melm.py:
´python melm_R2 -tr dataset/classification/diabetes_train -ts dataset/classification/diabetes_test -ty 1 -nh 100 -af dilation´

para melm_R2.py:
´python melm_R2 -tr dataset/classification/diabetes_train -ts dataset/classification/diabetes_test -ty 1 -nh 100 -af dilation´

para melm_R3.py:
´python melm_R3.py -tr benignos_v2.csv -ts malignos_v2.csv -ty 1 -nh 100 -af dilation´

## Limitações até o momento
Uma limitação fundamental do sistema sendo desenvolvido é que ele apenas funcionará com bases de dados binárias, onde há apenas uma classe e uma contra-classe, 1s e 0s.