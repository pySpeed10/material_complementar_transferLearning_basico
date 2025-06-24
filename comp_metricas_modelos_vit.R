# Instalar o pacote, se necessário
# install.packages("viridis")

library(ggplot2)
library(viridis)

# Dados conforme sua tabela
resultados <- data.frame(
  Modelo = rep(c(
    "ViT-1k (LR=0.1)", "ViT-21k (LR=0.1)",
    "ViT-1k (LR=0.01)", "ViT-21k (LR=0.01)",
    "ViT-1k (LR=0.001)", "ViT-21k (LR=0.001)"
  ), each = 5),
  Metrica = factor(rep(c("Acurácia", "Especificidade", "Sensibilidade", "Precisão", "F1-Score"), times = 6),
                   levels = c("Acurácia", "Especificidade", "Sensibilidade", "Precisão", "F1-Score")),
  Valor = c(
    0.3470, 0.6667, 0.3333, 0.1157, 0.1717,
    0.3470, 0.6667, 0.3333, 0.1157, 0.1717,
    0.4202, 0.7070, 0.4224, 0.4248, 0.3900,
    0.4068, 0.6992, 0.4070, 0.4079, 0.3889,
    0.5130, 0.7584, 0.5361, 0.5106, 0.5076,
    0.5100, 0.7572, 0.5336, 0.5064, 0.5015
  )
)

# Gráfico com paleta viridis
ggplot(resultados, aes(x = Metrica, y = Valor, fill = Modelo)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  scale_fill_viridis_d(option = "D") +  # Paleta viridis discreta
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
