# Vetores por fold — ResNet50
resnet_acc    <- c(0.6964, 0.5612, 0.7052, 0.7161, 0.6901)
resnet_recall <- c(0.7029, 0.5913, 0.7157, 0.7243, 0.6959)
resnet_prec   <- c(0.7066, 0.5685, 0.7183, 0.7237, 0.6986)
resnet_f1     <- c(0.7007, 0.5474, 0.7049, 0.7240, 0.6971)
resnet_esp    <- c(0.8450, 0.7849, 0.8504, 0.8550, 0.8417)

# Vetores por fold — ViT (ImageNet-1k)
vit_acc    <- c(0.4946, 0.4941, 0.4992, 0.5250, 0.5522)
vit_recall <- c(0.5191, 0.5137, 0.5225, 0.5540, 0.5712)
vit_prec   <- c(0.5009, 0.4985, 0.4970, 0.5408, 0.5507)
vit_f1     <- c(0.4865, 0.4227, 0.4960, 0.4911, 0.5513)
vit_esp    <- c(0.7507, 0.7489, 0.7503, 0.7654, 0.7767)

# Vetores por fold — ViT (IM-21k)
vit21k_acc    <- c(0.5178, 0.5075, 0.5214, 0.4951, 0.5083)
vit21k_recall <- c(0.5360, 0.5218, 0.5456, 0.5282, 0.5362)
vit21k_prec   <- c(0.5529, 0.5099, 0.5201, 0.3359, 0.5317)
vit21k_f1     <- c(0.4475, 0.4323, 0.5145, 0.4104, 0.4936)
vit21k_esp    <- c(0.7596, 0.7524, 0.7631, 0.7509, 0.7601)

vit21k <- data.frame(
  Acuracia = vit21k_acc,
  Recall = vit21k_recall,
  Precisao = vit21k_prec,
  F1 = vit21k_f1,
  Especificidade = vit21k_esp
)

# Agrupando em data.frames
resnet <- data.frame(Acuracia = resnet_acc, Recall = resnet_recall, Precisao = resnet_prec, F1 = resnet_f1, Especificidade = resnet_esp)
vit    <- data.frame(Acuracia = vit_acc,    Recall = vit_recall,    Precisao = vit_prec,    F1 = vit_f1,    Especificidade = vit_esp)

# Função para teste t pareado
comparar_modelos <- function(m1, m2, nome1, nome2) {
  resultado <- data.frame()
  for (metrica in colnames(m1)) {
    x <- m1[[metrica]]
    y <- m2[[metrica]]
    if (sd(x - y) == 0) {
      p <- NA
      status <- "Dados constantes"
    } else {
      teste <- t.test(x, y, paired = TRUE, alternative = "greater")
      p <- round(teste$p.value, 5)
      status <- ifelse(p < 0.05, "Sim", "Não")
    }
    resultado <- rbind(resultado, data.frame(
      Metrica = metrica,
      Comparação = paste(nome1, "vs", nome2),
      `p.valor` = p,
      `Significativo (α=0.05)` = status
    ))
  }
  return(resultado)
}

# Comparação: ResNet50 vs ViT (IM-1k)
tabela_resultado <- comparar_modelos(resnet, vit, "ResNet50", "ViT (IM-1k)")

# Exibir tabela
print(tabela_resultado, row.names = FALSE)

# Comparação ResNet50 vs ViT (IM-21k)
tabela_resultado_21k <- comparar_modelos(resnet, vit21k, "ResNet50", "ViT (IM-21k)")
#print(tabela_resultado_21k, row.names = FALSE)

# Comparação ViT (IM-1k) vs ViT (IM-21k)
tabela_resultado_vit <- comparar_modelos(vit, vit21k, "ViT (IM-1k)", "ViT (IM-21k)")
#print(tabela_resultado_vit, row.names = FALSE)

