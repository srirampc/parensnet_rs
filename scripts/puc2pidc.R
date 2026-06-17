library(argparser, quietly = TRUE)
library(rhdf5)
library(MASS)

load_puc <- function(fname) {
    h5ptr <- H5Fopen(fname)
    puc_data <- h5ptr$"/data"
    H5Fclose(h5ptr)

    puc_values <- puc_data$puc
    puc_index <- puc_data$index
    ngenes <- max(puc_index) + 1
    puc_scores <- matrix(0.0, ngenes, ngenes)
    nedges <- dim(puc_index)[2]
    for (jx in 1:nedges) {
        row_id <- puc_index[1, jx] + 1
        col_id <- puc_index[2, jx] + 1
        puc_scores[row_id, col_id] <- puc_values[jx]
        puc_scores[col_id, row_id] <- puc_values[jx]
    }
    list(index = puc_index, values = puc_values, scores = puc_scores)
}

puc2pidc <- function(puc_scores) {
    ngenes <- dim(puc_scores)[1]
    pidc_scores <- matrix(0.0, ngenes, ngenes)
    fit_list <- lapply(
        1:ngenes,
        function(ix) fitdistr(puc_scores[-ix, ix], "gamma")
    )
    cat(Sys.time(), "Build Fit List ", ngenes, "\n")
    for (ix in 1:(ngenes - 1)) {
        fit_i <- fit_list[[ix]]
        for (jx in (ix + 1):ngenes) {
            fit_j <- fit_list[[jx]]
            score <- puc_scores[ix, jx]
            score_i <- dgamma(
                score,
                shape = fit_i$estimate["shape"],
                scale = fit_i$estimate["scale"]
            )
            score_j <- dgamma(
                score,
                shape = fit_j$estimate["shape"],
                scale = fit_j$estimate["scale"]
            )
            score_i <- if (is.na(score_i) || is.infinite(score_i)) {
                0.0
            } else {
                score_i
            }
            score_j <- if (is.na(score_j) || is.infinite(score_j)) {
                0.0
            } else {
                score_j
            }

            pidc_scores[ix, jx] <- score_i + score_j
            pidc_scores[jx, ix] <- pidc_scores[ix, jx]
        }
    }
    pidc_scores
}

write_pidc <- function(puc_data, pidc_scores, pidc_file) {
    # Check its existence
    if (file.exists(pidc_file)) {
        # Delete file if it exists
        file.remove(pidc_file)
    }
    pidc_index <- puc_data$index
    pidc_values <- puc_data$puc
    nedges <- dim(pidc_index)[2]
    for (jx in 1:nedges) {
        row_id <- pidc_index[1, jx] + 1
        col_id <- pidc_index[2, jx] + 1
        pidc_values[jx] <- pidc_scores[row_id, col_id]
    }
    h5createFile(pidc_file)
    h5createGroup(pidc_file, "data")
    h5write(pidc_index, pidc_file, "data/index")
    h5write(pidc_values, pidc_file, "data/pidc")
}

main <- function(puc_file, pidc_file) {
    cat(Sys.time(), "Loading data from ", puc_file, "\n")
    puc_data <- load_puc(puc_file)
    cat(Sys.time(), "Loaded data from ", puc_file, "\n")
    pidc_scores <- puc2pidc(puc_data$scores)
    cat(Sys.time(), "Built puc scores ", puc_file, "\n")
    write_pidc(puc_data, pidc_scores, pidc_file)
    cat(Sys.time(), "Complete file ", puc_file, "\n")
}


# Create a parser
p <- arg_parser("Convert PUC 2 PIC")
p <- add_argument(p, "puc_file", help = "PUC file", type = "character")
p <- add_argument(p, "pidc_file", help = "PIDC file", type = "character")
# Parse the command line arguments
argv <- parse_args(p)

main(argv$puc_file, argv$pidc_file)
