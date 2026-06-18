options(warn=-1)
args = commandArgs(trailingOnly=TRUE)
var = args[1]
inpath = args[2]
outpath = args[2]
min = as.numeric(args[3])
max = as.numeric(args[4])
low = args[5]
high = args[6]
atlas = args[7]

library(ggplot2)
library(ggseg)
library(ggsegDKT)


# ─── FILTER ATLAS HELPERS ───

dkt_left <- dkt()
dkt_left$data$sf <- dkt_left$data$sf[
  grepl("^lh_", dkt_left$data$sf$label), ]

dkt_right <- dkt()
dkt_right$data$sf <- dkt_right$data$sf[
  grepl("^rh_", dkt_right$data$sf$label), ]


# ─── SUBCORTICAL: CORONAL ───

datar <- read.csv(paste(inpath, var, "_asegr.csv", sep=""), header=TRUE)

pdf(paste(outpath, var, "_aseg_coronal.pdf", sep=""))
print(
  ggplot(datar) +
    geom_brain(atlas = aseg(),
               aes(fill = r),
               colour = "grey",
               view = "coronal_1") +
    theme_void() +
    scale_fill_gradient(na.value = "lightgrey", low = low,
                        high = high, limits = c(min, max)) +
    labs(title = "", fill = "") +
    theme(legend.position = "none")
)
dev.off()


# ─── SUBCORTICAL: SAGITTAL ───

datal <- read.csv(paste(inpath, var, "_asegl.csv", sep=""), header=TRUE)

pdf(paste(outpath, var, "_aseg_sagittal.pdf", sep=""))
print(
  ggplot(datal) +
    geom_brain(atlas = aseg(),
               aes(fill = r),
               colour = "grey",
               view = "sagittal") +
    theme_void() +
    scale_fill_gradient(na.value = "lightgrey", low = low,
                        high = high, limits = c(min, max)) +
    labs(title = "", fill = "") +
    theme(legend.position = "none")
)
dev.off()


# ─── CORTICAL: 4 PANELS ───

data_lh <- read.csv(paste(inpath, var, "_cx_lh.csv", sep=""), header=TRUE)
data_rh <- read.csv(paste(inpath, var, "_cx_rh.csv", sep=""), header=TRUE)


# ─── LEFT LATERAL ───

pdf(paste(outpath, var, "_cx_left_lateral.pdf", sep=""))
print(
  ggplot(data_lh) +
    geom_brain(atlas = dkt_left,
               aes(fill = r),
               colour = "black",
               view = "lateral") +
    theme_void() +
    scale_fill_gradient(na.value = "lightgrey", low = low,
                        high = high, limits = c(min, max)) +
    labs(title = "", fill = "") +
    theme(legend.position = "none")
)
dev.off()


# ─── LEFT MEDIAL ───

pdf(paste(outpath, var, "_cx_left_medial.pdf", sep=""))
print(
  ggplot(data_lh) +
    geom_brain(atlas = dkt_left,
               aes(fill = r),
               colour = "black",
               view = "medial") +
    theme_void() +
    scale_fill_gradient(na.value = "lightgrey", low = low,
                        high = high, limits = c(min, max)) +
    labs(title = "", fill = "") +
    theme(legend.position = "none")
)
dev.off()


# ─── RIGHT LATERAL ───

pdf(paste(outpath, var, "_cx_right_lateral.pdf", sep=""))
print(
  ggplot(data_rh) +
    geom_brain(atlas = dkt_right,
               aes(fill = r),
               colour = "black",
               view = "lateral") +
    theme_void() +
    scale_fill_gradient(na.value = "lightgrey", low = low,
                        high = high, limits = c(min, max)) +
    labs(title = "", fill = "") +
    theme(legend.position = "none")
)
dev.off()


# ─── RIGHT MEDIAL ───

pdf(paste(outpath, var, "_cx_right_medial.pdf", sep=""))
print(
  ggplot(data_rh) +
    geom_brain(atlas = dkt_right,
               aes(fill = r),
               colour = "black",
               view = "medial") +
    theme_void() +
    scale_fill_gradient(na.value = "lightgrey", low = low,
                        high = high, limits = c(min, max)) +
    labs(title = "", fill = "") +
    theme(legend.position = "none")
)
dev.off()