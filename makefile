todo: main
main: main.cpp
	g++-4.9 -fdiagnostics-color=auto -std=c++11 -pthread -g -o main main.cpp \
		-Ilibarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp
clean:
	rm -f main tags
