# Instalar os pacotes, se necessário
# install.packages("ggplot2")
# install.packages("viridis")

library(ggplot2)
library(viridis)

# Criar o dataframe com os dados da sua tabela
resultados_cnn <- data.frame(
  Modelo = rep(c(
    "ResNet50 (LR=0.1)",
    "ResNet50 (LR=0.01)",
    "ResNet50 (LR=0.001)"
  ), each = 5),
  
  Metrica = factor(rep(c("Acurácia", "Especificidade", "Sensibilidade", "Precisão", "F1-Score"), times = 3),
                   levels = c("Acurácia", "Especificidade", "Sensibilidade", "Precisão", "F1-Score")),
  
  Valor = c(
    0.5863, 0.7909, 0.6026, 0.6186, 0.5856,     # LR=0.1
    0.6738, 0.8354, 0.6860, 0.6747, 0.6770,     # LR=0.01
    0.6595, 0.8274, 0.6662, 0.6629, 0.6644      # LR=0.001
  )
)

# Plotar gráfico com ggplot2 e viridis
ggplot(resultados_cnn, aes(x = Metrica, y = Valor, fill = Modelo)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "C") +  # outras opções: "C", "E"
  labs(
    x = "",
    y = "",
    fill = "Modelo"
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )
