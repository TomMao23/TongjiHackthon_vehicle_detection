#!/bin/sh
#echo 'Suggest run with root authority'
echo 'At runtime you should confirm that one of cuda 8.0/9.0/9.2/10 and cudnn have been installed.
run build.sh it will install mxnet-cu80/90/92/100/101mkl==1.5.0(depending on your cuda version) and gluoncv==0.5.0'
echo 'Confirmed? (y/n)'
read Confirmed

if [ "$Confirmed" = "y" ] || [ "$Confirmed" = "Y" ]; then
    echo 'Now MxNet 1.5.0 & gluoncv 0.5.0 would be installed!'
    echo
    echo 'Type in the version of your cuda (80/90/92/100/101)'
    read version
    
    if [ "$version" = "80" ]; then
        python -m pip install mxnet-cu80mkl==1.5.0
   
    elif [ "$version" = "90" ]; then
        python -m pip install mxnet-cu90mkl==1.5.0
    
    elif [ "$version" = "92" ]; then
        python -m pip install mxnet-cu92mkl==1.5.0
    
    elif [ "$version" = "100" ]; then
        python -m pip install mxnet-cu100mkl==1.5.0
    
    elif [ "$version" = "101" ]; then
        python -m pip install mxnet-cu101mkl==1.5.0
        
    #elif [ "$version" = "test" ]; then
       # pip install xlwt
    
    else 
        echo 'Unsupported version.'
        exit 1
    fi
    
    python -m pip install gluoncv==0.5.0
    
else
    echo 'Terminated.'
    exit 1
fi

echo 'Finished.'

