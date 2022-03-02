# Entropy estimation functions
print("in func_entropy_v1.R")
#print(getwd())
#print(dir("./"))
#library(FNN) # knn.dist
library(fields) # rdist
library(pracma) # Toeplitz(a, b)
library(R.matlab)
dyn.load("./kdpee/kdpeer.so") #kdpeer in Shannon_KDP
# Entropy functions

# Notes
# Funciones de entropia faltan 2: "Shannon_KDP",  "Shannon_vME"

# Falta agregar referencia ITE-package y referencia original a cada función

# Falta benchmarkear version R contra matlab en terminos de tiempo


# univariate functions

Shannon_1sp <- function(x){
  # From Mooij et al. 2016 pg 26
  res <- sort(x)
  # remove duplicates
  res <- unique(res)
  n <- length(res)
  res <- sum(log(abs(res[2:n]-res[1:(n-1)])))
  res <- res/(n-1)
  res <- res + digamma(n) - digamma(1)
  return(res)
}

Shannon_Gauss <- function(x){
  res <- log(var(x))
  #res <- 0.5(log(2*pi*exp(1))+res)
  return(res)
}

Shannon_spacing_V <- function(x){
  
  n <- length(x)
  m <- floor(sqrt(n))
  x_sorted <- sort(x)
  x_sorted <- c(rep(x_sorted[1], m), x_sorted, rep(x_sorted[n], m))
  diffs <- x_sorted[(2*m+1):(n+2*m)] - x_sorted[1:n]
  H = mean(log (n / (2*m) * diffs))
  return(H)
}

Shannon_spacing_Vb <- function(x){
  n <- length(x)
  m <- floor(sqrt(n))
  x_sorted <- sort(x)
  
  A <- x_sorted[(1+m):n]
  B <- x_sorted[1:(n-m)]
  
  diffs <- A - B;
  
  b <- sum(1/(m:n)) + log(m/(n+1)) # bias correction
  
  H <- mean(log((n+1)/m*diffs)) + b
  return(H)
}

Shannon_spacing_Vpconst <- function(x){
  
  n <- length(x)
  m <- floor(sqrt(n))
  x_sorted <- sort(x)
  x_sorted <- c(rep(x_sorted[1], m), x_sorted, rep(x_sorted[n], m))
  differs <- x_sorted[(2*m+1):(n+2*m)] - x_sorted[1:n]
  c <- c(rep(1,m), rep(2, n-2*m), rep(1,m)) # piecewise constant correction
  H <- mean(log (n / m * differs/c))
  return(H)
}

Shannon_spacing_Vplin <- function(x){
  n <- length(x)
  m <- floor(sqrt(n))
  x_sorted <- sort(x)
  x_sorted <- c(rep(x_sorted[1], m), x_sorted, rep(x_sorted[n],m))
  diffs <- x_sorted[(2*m+1):(n+2*m)] - x_sorted[1:n]
  C <- c(1+((1:m)-1)/m, rep(2,n-2*m), 1+(n-((n-m+1):n))/m) # piecewise linear correction
  H <- mean(log (n/ m * diffs/C))
  return(H)
}

Shannon_spacing_Vplin2 <- function(x){
  n <- length(x)
  m <- floor(sqrt(n)) 
  x_sorted <- sort(x);
  x_sorted <- c(rep(x_sorted[1],m), x_sorted, rep(x_sorted[n],m)) #%with the smallest (left) and largest (right) element
  diffs <- x_sorted[(2*m+1):(n+2*m)] - x_sorted[1:n]
  c1 <- 1 + ((1:m)+1)/m - (1:m)/(m^2)
  c2 <- rep(2,n-2*m-1)
  c3 <- 1 + (n-((n-m):n))/(m+1)
  C <- c(c1,c2,c3)
  H <- mean(log (n / m * diffs/C))
  return(H)
}

Shannon_spacing_VKDE <- function(x){
  #x <- seq(16)
  n <- length(x)
  m <- floor(sqrt(n))
  x_sorted <- sort(x)
  
  stdX <- sd(x) # sample standard deviation
  h <- 1.06 * stdX * n^(-1/5)
  A <- x_sorted[c(1:m, (n-m+1):n)]/h 
  B <- x_sorted/h
  sD <- fields:::rdist(A,B)^2 #squared distances between each real i.e. distance corresponds  absolute value
  
  s13 <- apply(exp(-sD/2),1, mean) / (sqrt(2*pi)*h) 
  s2 <- (2*m/n) / (x_sorted[(2*m+1):n] - x_sorted[1:(n-2*m)])
  
  H <- -mean(log(c(s13, s2)))
  return(H)
}

Shannon_spacing_LL <- function(x){
  n <- length(x)
  m <- floor(sqrt(n))
  x_sorted <- sort(x)
  x_sorted <- c(rep(x_sorted[1],m), x_sorted, rep(x_sorted[n],m))
  
  
  # we need rolling indices which are padded when we are near the edges of a vector and 
  # there are not enough values to the left or to the right
  
  # even numbered vectors shd be centered on the left of the two central indices ex: if winLen = 6
  # then for vector x = c(1, 2 ,3, 4, 5, 6, 7, 8, 9, 10) the rolling windows would be
  
  # c(0, 0, 1,  2,  3,  4 ); center <- 1
  # c(0, 1, 2,  3,  4,  5 ); center <- 2
  # c(1, 2, 3,  4,  5,  6 ); center <- 3
  # c(2, 3, 4,  5,  6,  7 ); center <- 4
  # c(3, 4, 5,  6,  7,  8 ); center <- 5
  # c(4, 5, 6,  7,  8,  9 ); center <- 6
  # c(5, 6, 7,  8,  9,  10); center <- 7
  # c(6, 7, 8,  9,  10, 0 ); center <- 8
  # c(7, 8, 9,  10, 0,  0 ); center <- 9
  # c(8, 9, 10, 0,  0,  0 ); center <- 10
  
  
  len <- length(x_sorted)
  winLen <- 2*m+1
  
  b <- sapply(1:length(x_sorted), function(i){
    indx <- (i-m):(i+m)
    indx[which(indx<=0 | indx> len)] <- NA
    res <- x_sorted[indx]
    res[which(is.na(res))] <- 0
    res <- res - mean(res)
    
    res <- as.numeric(t(-m:m) %*% res) / sum(res^2)
    
    return(res)
  })
  
  
  # b <- rep(0, len)
  # 
  # B = matrix(0, winLen, len)
  # 
  # for(i in 1:len){
  #   print(i)
  #   mid <- i + ((winLen+1)%%2)/2 
  #   CEIL <- ceiling(mid-floor(winLen/2))
  #   padLeft <- sum((CEIL:1)<=0)
  #   ini <- max(1, CEIL)
  #   FLOOR <- floor(mid + floor(winLen/2))
  #   padRight <- sum(((len):FLOOR)>len)
  #   fin <- min(len, FLOOR)
  #   z <- x_sorted[ini:fin]
  #   z <- c(rep(0, padLeft), z, rep(0, padRight)) 
  #   B[,i] <- z
  # }
  #     
  # 
  # for(i in 1:len){
  #   aux <- B[,i] - mean(B[,i])
  #   w <- (t(-m:m) %*% aux) / sum(aux^2)
  #   b[i] <- w
  # }
  
  
  
  b <- b[(m+1):(len-m)]
  H <- -mean(log(b/n))
  return(H)
  
}

Shannon_PSD_SzegoT <- function(x){
  #x <- 1:16
  p_max <- 10 
  K <- 100 
  
  
  a <- 1/(2*max(abs(x)))
  
  
  x <- x*a
  
  ###################################################
  #        Estimation of the AR parameters          #
  ###################################################
  
  #all_a_AR <- matrix(0, p_max-1,p_max)
  #all_e <- rep(0,p_max-1)
  
  all_a_AR <- matrix(0, p_max, p_max +1)
  all_e <- rep(0,p_max)
  
  # i is complex number so phi_x is too
  phi_x <- apply(exp(1i*2*pi*(0:p_max)%*%t(x)),1, mean)
  
  for (order in 1:p_max){       
    # LONG AR
    # complex number matrix
    # order <- p_max
    #print(order)
    
    
    if(order== 1){
      R <- matrix(phi_x[1], 1, 1)
    } else{
      R <- Toeplitz(phi_x[1:order], Conj(phi_x[1:order]))
    }
    r <- phi_x[2:(order+1)]
    Delta <- diag((1:order)^4)
    aux <- -(R + (1e-5)*Delta)
    
    aux2 <- solve(aux, r)
    
    a_AR <- c(1 , aux2)    
    all_a_AR[order, 1:(order+1)] <- a_AR
    all_e[order] <- as.numeric(Re(t(a_AR) %*% Conj(phi_x[1:(order+1)])))
    #print(paste("all_e[order]", all_e[order]))
  }
  # all_aAR is complex
  # all_e is real
  
  MDL <- length(x)*log(all_e) + (1:length(all_e))*log(length(x))
  
  p <- which.min(MDL)
  min_value <- MDL[p]
  
  a_AR <- all_a_AR[p,1:(p+1)]
  
  ###################################################
  #     Extrapolation of the autocorrelation        #
  ###################################################
  
  
  L <- K*(p+1)
  
  phi_x_tilde <- rep(0, L)
  phi_x_tilde[1:(p+1)] <- phi_x[1:(p+1)]
  
  for(kk in (p+2):L){
    # kk <- 20
    phi_x_tilde[kk] <- - as.complex(t(a_AR[2:length(a_AR)]) %*% phi_x_tilde[(kk-1):(kk-p)])
  }
  
  R <- Toeplitz(phi_x_tilde, Conj(phi_x_tilde))
  
  ######################################################
  #  Estimation of the Entropy using Szeg�'s Theorem   #
  ######################################################
  
  lambdas <- pracma:::eig(R)
  
  H <- -mean(Re(log2(as.complex(lambdas))*lambdas))
  
  #H2 <- H - log2(a)
  #H2 <- H2 / log2(exp(1))
  #H3 <- H/log2(exp(1)) - log(a)
  #H4 <- log(2^H) - log(a)
  
  H <- log(2^H / a)
  
  #H; H2; H3; H4
  
  return(H)
  
}

Shannon_MaxEnt1 <- function(x){
  n <- length(x)
  
  # normalize Y to have zero mean and unit std:
  x <- x - mean(x) # E=0, this step does not change the Shannon entropy of the variable
  
  s <- sqrt(sum(x^2)/(n-1))
  x <- x/s
  
  H_whiten <- log(s) #we will take this scaling into account via the entropy transformation rule [ H(wz) = H(z)+log(|w|) ] at the end
  
  # H1,H2 -> H:
  H1 <- ( 1+log(2*pi) ) / 2 # %=H[N(0,1)]
  
  #H2:
  k1 <- 36 / ( 8*sqrt(3) - 9 )
  k2a <- 1 / ( 2 - 6/pi )
  H2 <- k1 * mean(x * exp(-x^2/2))^2 + k2a * (mean(abs(x)) - sqrt(2/pi))^2
  H <- H1 - H2
  
  #take into account the 'std=1' pre-processing:
  H <- H + H_whiten
  return(H)
}

Shannon_MaxEnt2 <- function(x){
  n <- length(x)
  
  # normalize Y to have zero mean and unit std:
  x <- x - mean(x) # E=0, this step does not change the Shannon entropy of the variable
  
  s <- sqrt(sum(x^2)/(n-1))
  x <- x/s
  
  H_whiten <- log(s) #we will take this scaling into account via the entropy transformation rule [ H(wz) = H(z)+log(|w|) ] at the end
  
  # H1,H2 -> H:
  H1 <- ( 1+log(2*pi) ) / 2 # %=H[N(0,1)]
  
  #H2:
  k1 <- 36 / ( 8*sqrt(3) - 9 )
  k2b <- 24 / (16*sqrt(3) - 27)
  H2 <- k1 * mean(x * exp(-x^2/2))^2 + k2b * (mean(exp(-x^2/2)) - sqrt(1/2))^2
  H <- H1 - H2
  
  #take into account the 'std=1' pre-processing:
  H <- H + H_whiten
  return(H)
}

# multivariate functions
# for now multivariate gaussian entropy
Shannon_expF <- function(x){
  # n <- 10000; d<- 10; x <- matrix(rnorm(n*d), n, d)
  if(class(x) != "matrix"){
    x <- matrix(x, length(x), 1)
  }
  n <- nrow(x)
  d <- ncol(x)
  m <- apply(x, 2, mean)
  C <- cov(x)
  invC <- solve(C)
  t1 <- as.numeric(invC %*% m)
  t2 <- invC / 2
  In <- solve(t2)
  term1 <- sum(diag(In %*% t1 %*% t(t1))) /4 - log(det(t2)) / 2 + d * log(pi) / 2
  
  
  s <- In %*% t1
  gradF.t1 <- s / 2
  gradF.t2 <- -In/2 - (s  %*% t(s)) / 4
  
  
  term2 <- sum(t1 * gradF.t1) + sum(t2 * gradF.t2)
  
  H <-  term1 - term2 #assumption: k, the carrier measure is zero
  return(H)
  
}

Shannon_vME <- function(x){
  # n <- 10000; d<- 10; x <- matrix(rnorm(n*d), n, d)
  if(class(x) != "matrix"){
    x <- matrix(x, length(x), 1)
  }
  n <- nrow(x)
  d <- ncol(x)
  stop("currently not implemented")
  
  return(H) 
}

Shannon_kNN_k <- function(x, k){
  if(class(x) != "matrix"){
    x <- matrix(x, length(x), 1)
  }
  d <- ncol(x)
  n <- nrow(x)
  squared_distances <- FNN:::knn.dist(x, k)^2
  V <- pi^(d/2) / gamma(d/2+1)
  H <-  log(n-1) - digamma(k) + log(V) + d / n * sum(log(sqrt(squared_distances[,k]))) 
  return(H)
  
}

Shannon_Edgeworth <- function(x){
  #x <- matrix(rnorm(10000*3), 10000, 10)
  #x <- matrix(rnorm(10000*1), 10000, 1)
  if(class(x) != "matrix"){
    x <- matrix(x, length(x), 1)
  }
  n <- nrow(x)
  d <- ncol(x)
  
  
  # normalize Y to have zero mean and unit std:
  x <- apply(x, 2, function(col) col-mean(col)) # E=0, this step does not change the Shannon entropy of the variable
  
  s <- sqrt(apply(x^2, 2, sum)/(n-1))
  x <- t(t(x) / s)
  
  
  
  H_whiten <- log(prod(s)) #we will take this scaling into account via the entropy transformation rule [ H(Wz) = H(z)+log(|det(W)|) ] at the end
  
  dete <- det(cov(x))
  #print(dete)
  dete <- max(dete, 1e-300)
  
  H_normal <- log(dete)/2 + d/2 * log(2*pi) + d/2 #Shannon entropy of a normal variable with cov(Y.') covariance.
  
  #t1b:
  # t1b <- 0
  # for(i in 1:d){ #d terms
  #   kappa_iii <- mean(x[,i]^3) 
  #   t1b <- t1b + kappa_iii^2
  # }
  
  t1 <- sum(apply(x^3, 2, mean)^2)
  #t1-t1b              
  
  #t2:
  # pm <- proc.time()
  # t2b <- 0
  # for(i in 1:d){
  #   for(j in setdiff(1:d,i)){ #j\ne i; 2*nchoosek(d,2) terms
  #     kappa_iij <- mean(x[,i]^2 * x[,j])
  #     t2b <- t2b + kappa_iij^2
  #   }
  # }
  # proc.time()-pm
  
  #pm <- proc.time()
  t2 <- sum(((t(x^2) %*% x/n)[diag(d)==0])^2) 
  #proc.time()-pm
  #t2-t2b
  
  t2 <- 3 * t2
  
  #t3:
  # pm <- proc.time()
  # t3b <- 0
  # for(i in 1:(d-2)){ #i<j<k; nchoosek(d,3) terms
  #   for(j in (i+1):(d-1)){
  #     for(k in (j+1):d){
  #       kappa_ijk <- mean(x[,i]*x[,j]*x[,k])
  #       t3b <- t3b + kappa_ijk^2
  #     }
  #   }
  # }
  # proc.time()-pm
  
  
  #pm <- proc.time()
  indx <- expand.grid(i=1:max(1,d-2), j=1:max(1, d-1), k=1:d)
  indx <- indx[which(indx$j>indx$i & indx$k>indx$j),]
  t3 <- sum(apply(indx, 1, function(row) mean(x[,row[1]]*x[,row[2]]*x[,row[3]])^2))
  #proc.time() - pm
  #t3-t3b
  
  
  
  
  t3 <- t3 / 6
  
  H <- (H_normal - (t1+t2+t3) / 12) + H_whiten
  return(H)
}

Shannon_KDP <- function(x) {
  if(class(x)!="matrix"){
    x <- matrix(x, length(x), 1)
  } 
  
  if(any(is.na(x))){
    stop("not messing with sending NAs to kdpee.c bruh")
  }
  
  n <- as.integer(nrow(x))
  d <- as.integer(ncol(x))
  mins <- as.double(apply(x, 2, min))
  maxs <- as.double(apply(x, 2, max))
  x <- as.double(as.numeric(x))
  zcut <- as.double(1.96)
  res <- .C("kdpeer", x, n, d, mins, maxs, zcut, numeric(1))
  return(res[[length(res)]])
}


Shannon_kNN_k_matlab <- function(x, matlabSession, k){
  if(class(x) != "matrix"){
    x <- matrix(x, length(x), 1)
  }
  
  
  # pass sample to matlab
  setVariable(matlabSession, x=x)
  #evaluate(matlab, "size(x)")
  
  # calculate entropy in matlab - this code varies by type of entropy
  evaluate(matlabSession, "mult=1;")
  parmInitString <- paste("co = HShannon_kNN_k_initialization(mult, {'k',",k,"});")
  evaluate(matlabSession, parmInitString)
  evaluate(matlabSession, "H = HShannon_kNN_k_estimation(x, co);")
  
  # Bring back to R
  res <- getVariable(matlabSession, c("H"))$H
  
  
  
  return(as.numeric(res))
}


genericMatlabEntropy <- function(x, matlabSession, type=c("Shannon_kNN_k", "Shannon_spacing_V", "Shannon_spacing_Vb", 
                                                          "Shannon_spacing_Vpconst","Shannon_spacing_Vplin", "Shannon_spacing_Vplin2", "Shannon_spacing_VKDE", 
                                                          "Shannon_spacing_LL", "Shannon_KDP", "Shannon_PSD_SzegoT", "Shannon_Edgeworth", "Shannon_MaxEnt1", 
                                                          "Shannon_MaxEnt2", "Shannon_expF", "Shannon_vME") , ...){
  
  pars <- list(...)
  if(class(x) != "matrix"){
    x <- matrix(x, 1, length(x))
  } else{
    x <- t(x)
  }
  
  p <- nrow(x)
  n <- ncol(x)
  
  # pass sample to matlab
  setVariable(matlabSession, x=x)
  #evaluate(matlab, "size(x)")
  
  # calculate entropy in matlab - this code varies by type of entropy
  evaluate(matlabSession, paste("mult=", p,";", sep=""))
  initFuncString <- paste("H", type, "_initialization", sep="")
  
  if(length(pars)>0){
    parmString <- paste("'", names(pars) ,"',", sapply(pars, function(el) el), sep="")
    parmString <- paste(",{", parmString, "}", sep="")  
    parmInitString <- paste("co = ", initFuncString,"(mult", parmString,");", sep="")
  } else{
    parmInitString <- paste("co = ", initFuncString,"(mult);", sep="")
  }
  evaluate(matlabSession, parmInitString)
  estFuncString <- paste("H", type, "_estimation", sep="")
  evaluate(matlabSession, "H = ", estFuncString,"(x, co);", sep="")
  
  # Bring back to R
  res <- getVariable(matlabSession, c("H"))$H
  
  return(as.numeric(res))
}

