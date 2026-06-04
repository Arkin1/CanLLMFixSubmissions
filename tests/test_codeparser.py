import unittest
from unittest.mock import MagicMock
from genai.methods import LLMMethod
from genai.utils import CodeParser
from utils import clean_source_code

class TestReward(unittest.TestCase):
    def test_code_parser_no_cpp(self):
        code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(code, parsed_code)

    def test_code_parser_with_cpp(self):
        code = '''```cpp
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)

    def test_code_parser_with_cpp_multiple_cpp_blocks(self):
        code = '''```cpp
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            dsfsdfsdf sdf sdfsd sdf sfds fdsf
            ```cpp
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)

    def test_code_parser_with_cplusplus(self):
        code = '''```c++
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)

    def test_code_parser_with_cplusplus_multiple_blocks(self):
        code = '''```c++
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            sdfsdfsd
            sdfsdfds
            ```c++
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)
    
    def test_code_parser_with_c(self):
        code = '''```c
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)

    def test_code_parser_with_c_multiple_blocks(self):
        code = '''```c
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            sdfdsf sdf sdf
            ```c
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)

    def test_code_parser_with_cpp(self):
        code = '''<think>
                  This is a C++ code snippet.
            </think>
        ```cpp
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)
        
    def test_code_parser_with_cpp_multiple_blocks(self):
        code = '''<think>
                  This is a C++ code snippet.
            </think>
        ```cpp
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
        sdfdsfdsfs
         dsf

         ```cpp
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            ```
            '''
        correct_code = '''
                    #include<iostream>
            using namespace std;
            int main() {
                    int s = 0;
                    int n = 10;
                    for(int i=1;i<n;++i)
                    {
                            s += i;
                    }
            cout << s << endl;
            return 0;
            }
            '''
        parsed_code = CodeParser.extract_code(code)

        self.assertEqual(correct_code, parsed_code)
        