setwd("C:/Users/Malin Janna Vega/Downloads/CONUS_MASTER_FOLDER")

if (!requireNamespace("geomorph", quietly = TRUE)) install.packages("geomorph")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")

library(geomorph)
library(dplyr)
library(tidyr)

fn <- "C:/Users/Malin Janna Vega/Downloads/CONUS_MASTER_FOLDER/landmarks_raw/Conus_striolatus_landmarks_cropped.csv"
df <- read.csv(fn, stringsAsFactors = FALSE, check.names = FALSE)

cat("Rows (specimens):", nrow(df), "\n")
cat("Columns:", ncol(df), "\n")
print(head(df))

#Detect x/y columns and sort them by landmark index
x_cols <- grep("^x\\d+$", names(df), value = TRUE)
y_cols <- grep("^y\\d+$", names(df), value = TRUE)

if (length(x_cols) == 0 | length(y_cols) == 0) {
  stop("Couldn't find columns named like x0,y0,... in the CSV. Check column names.")
}

get_index <- function(nm) as.numeric(sub("^[xy]", "", nm))
x_cols <- x_cols[order(get_index(x_cols))]
y_cols <- y_cols[order(get_index(y_cols))]

p <- length(x_cols)            #number of landmarks
n <- nrow(df)                  #number of specimens
cat("Detected", p, "landmarks and", n, "specimens\n")

#Build the coordinates array: p × k × n (k=2 for 2D)
coords_array <- array(NA_real_, dim = c(p, 2, n))

for (i in seq_len(n)) {
  xs <- as.numeric(df[i, x_cols])
  ys <- as.numeric(df[i, y_cols])
  coords_array[, 1, i] <- xs
  coords_array[, 2, i] <- ys
}

#Check for missing coordinates
if (any(is.na(coords_array))) {
  stop("There are missing coordinates (NA) in the data. Fix or impute them before GPA.")
}

#Run GPA (geomorph)
#By default gpagen centers, rotates and scales shapes (scale = TRUE).
#If you want to keep size information (no scaling), set scale = FALSE.
gpa <- gpagen(coords_array, print.progress = TRUE) 

gpa$coords[, 1, ] <- -gpa$coords[, 1, ]
gpa$coords[, 2, ] <- -gpa$coords[, 2, ]
gpa$consensus[, 1] <- -gpa$consensus[, 1]
gpa$consensus[, 2] <- -gpa$consensus[, 2]

#gpa$coords is the aligned coordinates (p x 2 x n)
#gpa$Csize is the centroid size (pre-scaling)
#gpa$consensus is the mean (consensus) shape (p x k matrix)

consensus <- gpa$consensus   # p x 2 matrix
write.csv(consensus, "Conus_striolatus_gpa_consensus.csv", row.names = FALSE)

specimens <- if ("image" %in% names(df)) df$image else paste0("specimen", seq_len(n))
csize <- gpa$Csize

long_list <- lapply(seq_len(n), function(i) {
  data.frame(
    specimen = specimens[i],
    landmark = 0:(p-1),
    x = gpa$coords[, 1, i],
    y = gpa$coords[, 2, i],
    centroid_size = csize[i],
    stringsAsFactors = FALSE
  )
})
long_df <- do.call(rbind, long_list)
write.csv(long_df, "Conus_striolatus_landmarks_cropped_gpa_long.csv", row.names = FALSE)

wide_df <- data.frame(specimen = specimens, centroid_size = csize, stringsAsFactors = FALSE)
for (j in seq_len(p)) {
  wide_df[[paste0("x", j-1)]] <- gpa$coords[j, 1, ]
  wide_df[[paste0("y", j-1)]] <- gpa$coords[j, 2, ]
}
write.csv(wide_df, "Conus_striolatus_landmarks_cropped_gpa_wide.csv", row.names = FALSE)

plotAllSpecimens(gpa$coords, mean = TRUE)

cat("Saved:\n - Conus_striolatus_landmarks_cropped_gpa_long.csv\n - Conus_striolatus_landmarks_cropped_gpa_wide.csv\n - Conus_striolatus_gpa_consensus.csv\n")