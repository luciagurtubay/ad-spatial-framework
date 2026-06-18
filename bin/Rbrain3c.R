options(warn=-1)
args = commandArgs(trailingOnly=TRUE)
var = args[1]
inpath = args[2]
outpath = args[2]
min = as.numeric(args[3])
midpoint = 0
max = as.numeric(args[4])
low = args[5]
mid = args[6]
high = args[7]
atlas = args[8]

library(ggplot2)
library(ggseg)
library(ggsegDKT)

hemisphere = c("left", "right")
view = c("lateral", "medial")


# ─── SUBCORTICAL (aseg) ───

datal <- read.csv(paste(inpath, var, "_asegl.csv", sep=""), header=TRUE)
datar <- read.csv(paste(inpath, var, "_asegr.csv", sep=""), header=TRUE)

pdf(paste(outpath, var, "_asegr.pdf", sep=""))
print(
  ggplot(datar) +
    geom_brain(atlas = aseg(), aes(fill = r), colour = "grey") +
    theme_void() +
    scale_fill_gradient2(na.value = "lightgrey", low = low,
                         mid = mid, high = high,
                         midpoint = midpoint,
                         limits = c(min, max)) +
    labs(title = "", fill = "")
)
dev.off()

pdf(paste(outpath, var, "_asegl.pdf", sep=""))
print(
  ggplot(datal) +
    geom_brain(atlas = aseg(), aes(fill = r), colour = "grey") +
    theme_void() +
    scale_fill_gradient2(na.value = "lightgrey", low = low,
                         mid = mid, high = high,
                         midpoint = midpoint,
                         limits = c(min, max)) +
    labs(title = "", fill = "")
)
dev.off()


# ─── CORTICAL (DKT) ───

data <- read.csv(paste(inpath, var, "_cx.csv", sep=""), header=TRUE)

for (h in hemisphere) {
  for (vi in view) {
    pdf(paste(outpath, var, "_cx_", h, "_", vi, ".pdf", sep=""))
    print(
      ggplot(data) +
        geom_brain(atlas = dkt(),
                   aes(fill = r),
                   colour = "black",
                   hemisphere = h,
                   view = vi) +
        theme_void() +
        scale_fill_gradient2(na.value = "lightgrey", low = low,
                             mid = mid, high = high,
                             midpoint = midpoint,
                             limits = c(min, max)) +
        labs(title = "", fill = "")
    )
    dev.off()
  }
}