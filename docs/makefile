all:
	doxygen Doxyfile
	make -C latex/
	ssh aisa 'rm -r public_html/mpi_wrapper'
	scp -r html aisa:public_html/mpi_wrapper
