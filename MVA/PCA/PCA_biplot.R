# 載入必要套件
library(tidyverse)
library(factoextra)
library(gridExtra)
library(ggrepel)

# 準備數據的函數
prepare_data <- function(df) {
  attitude_cols <- c(
    paste0("q22_0", 1:5, "_1"),
    paste0("q23_0", 1:5, "_1"),
    paste0("q25_0", 1:4, "_1"),
    paste0("q26_0", 1:3, "_1")
  )
  
  X <- df[, attitude_cols]
  X <- na.omit(X)
  
  return(list(X = X, attitude_cols = attitude_cols))
}

# 執行PCA的函數
do_pca <- function(X) {
  pca_result <- prcomp(X, scale. = TRUE)
  return(pca_result)
}

# 分析主成分的函數
analyze_components <- function(pca_result, loadings, attitude_cols) {
  n_components <- 4
  
  # 輸出總解釋變異量
  var_explained <- summary(pca_result)$importance[2, ]
  cum_var_explained <- summary(pca_result)$importance[3, ]
  
  cat("\nPrincipal Components Analysis Results:\n")
  cat("========================================\n")
  
  cat(sprintf("\nTotal variance explained by first %d components: %.2f%%\n",
              n_components, cum_var_explained[n_components] * 100))
  
  # 分析每個主成分
  for(i in 1:n_components) {
    cat(sprintf("\nPC%d (Variance explained: %.2f%%)\n", i, var_explained[i] * 100))
    cat("----------------------------------------\n")
    
    loadings_pc <- loadings[, i]
    names(loadings_pc) <- attitude_cols
    
    # 顯示最重要的正向和負向loading
    cat("Most important positive loadings:\n")
    print(sort(loadings_pc[loadings_pc > 0.3], decreasing = TRUE)[1:3])
    
    cat("\nMost important negative loadings:\n")
    print(sort(loadings_pc[loadings_pc < -0.3])[1:3])
  }
}

# 繪製PCA相關性的函數
plot_pca_correlations <- function(scores) {
  # 只取前4個主成分
  scores_df <- as.data.frame(scores[, 1:4])
  colnames(scores_df) <- paste0("PC", 1:4)
  
  # 繪製相關矩陣
  cor_matrix <- cor(scores_df)
  
  # 使用ggplot2繪製熱力圖
  cor_long <- cor_matrix %>%
    as.data.frame() %>%
    rownames_to_column("Var1") %>%
    pivot_longer(-Var1, names_to = "Var2", values_to = "Correlation")
  
  ggplot(cor_long, aes(Var1, Var2, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
    geom_text(aes(label = round(Correlation, 3)), size = 3) +
    theme_minimal() +
    labs(title = "Principal Components Correlation Matrix")
  
  # 繪製散點圖矩陣
  pairs(scores_df, main = "Pairwise Scatter Plots of Principal Components")
  
  return(cor_matrix)
}

# 繪製PCA向量圖函數
plot_pca_vectors <- function(pca_result) {
  # 準備loadings
  loadings <- pca_result$rotation
  
  # 定義問題群組和顏色
  question_groups <- list(
    q22 = list(
      vars = paste0("q22_0", 1:5, "_1"),
      color = "#FF6B6B"  # 紅色系
    ),
    q23 = list(
      vars = paste0("q23_0", 1:5, "_1"),
      color = "#4ECDC4"  # 青色系
    ),
    q25 = list(
      vars = paste0("q25_0", 1:4, "_1"),
      color = "#45B7D1"  # 藍色系
    ),
    q26 = list(
      vars = paste0("q26_0", 1:3, "_1"),
      color = "#96CEB4"  # 綠色系
    )
  )
  
  # 定義PC組合
  pc_pairs <- list(
    c(1,2), c(1,3), c(1,4),
    c(2,3), c(2,4), c(3,4)
  )
  
  # 為每個PC組合創建單獨的圖
  for(pair in pc_pairs) {
    pc1 <- pair[1]
    pc2 <- pair[2]
    
    # 創建新的圖形窗口
    dev.new(width = 8, height = 8)
    
    # 設置繪圖參數
    par(mar = c(5, 5, 4, 2), 
        family = "serif")
    
    # 創建空白圖
    plot(0, 0, type = "n", 
         xlim = c(-1, 1), 
         ylim = c(-1, 1),
         xaxt = "n", 
         yaxt = "n", 
         xlab = paste("PC", pc1),
         ylab = paste("PC", pc2),
         main = paste("PC", pc1, "vs PC", pc2),
         cex.main = 1.2,
         cex.lab = 1.1,
         asp = 1)
    
    # 添加網格線
    grid(nx = 10, ny = 10, col = "gray92", lty = 1)
    
    # 添加參考圓
    symbols(0, 0, circles = 1, inches = FALSE, add = TRUE, 
            lty = 2, fg = "gray60")
    
    # 添加座標軸
    abline(h = 0, v = 0, col = "gray70", lwd = 1)
    
    # 自定義座標軸刻度
    at <- c(-1.0, -0.5, 0.0, 0.5, 1.0)
    labels <- c("-1.0", "-0.5", "0.0", "0.5", "1.0")
    
    # 添加座標軸和刻度
    axis(1, at = at, labels = labels, cex.axis = 0.9)
    axis(2, at = at, labels = labels, cex.axis = 0.9, las = 1)
    
    # 為每個問題群組繪製向量
    for(group in question_groups) {
      group_vars <- group$vars
      group_color <- group$color
      
      arrows(0, 0, 
             loadings[group_vars, pc1], 
             loadings[group_vars, pc2],
             length = 0.12,
             angle = 15,
             col = group_color,
             lwd = 2.2)
      
      vec_length <- sqrt(loadings[group_vars, pc1]^2 + loadings[group_vars, pc2]^2)
      important_vars <- vec_length > quantile(vec_length, 0.3)
      
      if(any(important_vars)) {
        text(loadings[group_vars[important_vars], pc1], 
             loadings[group_vars[important_vars], pc2], 
             group_vars[important_vars],
             pos = 4,
             offset = 0.5,
             cex = 0.9,
             col = group_color,
             font = 2)
      }
    }
  }
}

# 繪製貢獻度圖函數
plot_contributions <- function(pca_result) {
  var_contrib <- get_pca_var(pca_result)$contrib
  total_contrib <- rowSums(var_contrib)
  
  contrib_df <- data.frame(
    Variable = rownames(var_contrib),
    Contribution = total_contrib
  )
  
  dev.new(width = 10, height = 6)
  print(
    ggplot(contrib_df, aes(x = reorder(Variable, -Contribution), y = Contribution)) +
      geom_bar(stat = "identity", fill = "#4ECDC4") +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        axis.text.y = element_text(size = 10),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14, face = "bold")
      ) +
      labs(title = "Total Variable Contributions",
           x = "Variables",
           y = "Contribution")
  )
}

# 主函數
main <- function() {
  # 讀取數據
  df <- read.csv("/Users/tommy/Desktop/應用多變量分析/processed_data_with_score.csv")
  
  # 準備數據
  data_list <- prepare_data(df)
  X <- data_list$X
  attitude_cols <- data_list$attitude_cols
  
  # 執行PCA
  pca_result <- do_pca(X)
  
  # 分析主成分
  loadings <- as.data.frame(pca_result$rotation[, 1:4])
  analyze_components(pca_result, loadings, attitude_cols)
  
  # 計算得分和相關性
  scores <- pca_result$x
  correlations <- plot_pca_correlations(scores)
  
  # 繪製向量圖
  plot_pca_vectors(pca_result)
  
  # 繪製貢獻度圖
  plot_contributions(pca_result)
  
  return(list(
    pca_result = pca_result,
    scores = scores,
    loadings = loadings,
    correlations = correlations
  ))
}

# 執行主函數
results <- main()

