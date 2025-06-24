# Instalar o pacote caso ainda não esteja instalado
#install.packages("ggplot2")

# Carregar a biblioteca
library(ggplot2)

# Criar dataframe com métricas na ordem desejada
resultados <- data.frame(
  Modelo = rep(c("ResNet50 (IM-1k)", "ViT (IM-1k)", "ViT (IM-21k)"), each = 5),
  Metrica = factor(rep(c("Acurácia", "Especificidade", "Sensibilidade", "Precisão", "F1-Score"), times = 3),
                   levels = c("Acurácia", "Especificidade", "Sensibilidade", "Precisão", "F1-Score")),
  Valor = c(
    0.6738, 0.8354, 0.6860, 0.6747, 0.6770,     # ResNet50 (melhor em LR=0.01)
    0.5130, 0.7584, 0.5361, 0.5106, 0.5076,     # ViT IM-1k (melhor em LR=0.001)
    0.5100, 0.7572, 0.5336, 0.5064, 0.5015      # ViT IM-21k (melhor em LR=0.001)
  )
)

# Plotar gráfico de barras comparativas
ggplot(resultados, aes(x = Metrica, y = Valor, fill = Modelo)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) +
  labs(
    #title = "Comparação de Métricas entre ResNet50 e ViT (IM-1k e IM-21k)",
    x = "",
    y = "",
    fill = "Modelo"
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  theme_minimal(base_size = 13)
