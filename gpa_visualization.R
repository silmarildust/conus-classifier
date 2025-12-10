setwd("C:/Users/Malin Janna Vega/Downloads/CONUS_MASTER_FOLDER")

if (!requireNamespace("geomorph", quietly = TRUE)) install.packages("geomorph")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")
if (!requireNamespace("abind", quietly = TRUE)) install.packages("abind")

library(geomorph)
library(dplyr)
library(tidyr)
library(abind)

read_coords <- function(filename) {
  df <- read.csv(filename, stringsAsFactors = FALSE, check.names = FALSE)
  
  x_cols <- grep("^x\\d+$", names(df), value = TRUE)
  y_cols <- grep("^y\\d+$", names(df), value = TRUE)
  x_cols <- x_cols[order(as.numeric(sub("^[xy]", "", x_cols)))]
  y_cols <- y_cols[order(as.numeric(sub("^[xy]", "", y_cols)))]
  
  p <- length(x_cols)
  n <- nrow(df)
  
  coords_array <- array(NA_real_, dim = c(p, 2, n))
  for (i in seq_len(n)) {
    coords_array[, 1, i] <- as.numeric(df[i, x_cols])
    coords_array[, 2, i] <- as.numeric(df[i, y_cols])
  }
  
  coords_array
}

coords_striolatus <- read_coords("landmarks_raw/Conus_striolatus_landmarks_cropped.csv")
coords_muriculatus <- read_coords("landmarks_raw/Conus_muriculatus_landmarks_cropped.csv")

coords_combined <- abind(coords_striolatus, coords_muriculatus, along = 3)

gpa_combined <- gpagen(coords_combined, print.progress = TRUE)

species <- c(rep("Striolatus", dim(coords_striolatus)[3]),
             rep("Muriculatus", dim(coords_muriculatus)[3]))

# align coordinate system
coords_rotated <- gpa_combined$coords
coords_rotated[, 1, ] <- -coords_rotated[, 1, ]  # flip x
coords_rotated[, 2, ] <- -coords_rotated[, 2, ]  # flip y

# plot overlay
plotAllSpecimens(coords_rotated, mean = FALSE)
for (i in 1:dim(coords_rotated)[3]) {
  col <- ifelse(species[i] == "Striolatus", "green", "pink")
  points(coords_rotated[, , i], col = col, pch = 16)
}

cons_striolatus <- apply(coords_rotated[, , species == "Striolatus"], 1:2, mean)
cons_muriculatus <- apply(coords_rotated[, , species == "Muriculatus"], 1:2, mean)

points(cons_striolatus, col = "darkgreen", pch = 19, cex = 2)
points(cons_muriculatus, col = "deeppink", pch = 19, cex = 2)

legend("topright",
       legend = c("Striolatus specimens", "Muriculatus specimens",
                  "Striolatus mean", "Muriculatus mean"),
       col = c("green", "pink", "darkgreen", "deeppink"),
       pch = c(16, 16, 19, 19),
       bty = "n")

title("Overlay")