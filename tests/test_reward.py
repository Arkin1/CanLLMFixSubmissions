import unittest
from unittest.mock import AsyncMock
from genai.methods import LLMMethod
import asyncio
from utils import clean_source_code

class TestReward(unittest.IsolatedAsyncioTestCase):
        async def test_reward_function_equal(self):
                buggy_code = '''
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
                correct_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                
                generated_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                mock_problem_manager = AsyncMock()
                mock_problem_manager.evaluate_submission.return_value = 1.0

                method = LLMMethod("vendor1", "model1", mock_problem_manager)

                buggy_code = clean_source_code(buggy_code)
                correct_code = clean_source_code(correct_code)

                reward = await method.compute_reward(problem_id = "problem1", 
                                        buggy_code=buggy_code,
                                        generated_code=generated_code, 
                                        correct_code=correct_code)
                
                self.assertEqual(reward.context.num_lines_buggy_solution, 12)
                self.assertEqual(reward.context.num_lines_correct_solution, 12)
                self.assertEqual(reward.context.num_lines_generated_solution, 12)

                self.assertEqual(reward.context.num_added_lines_baseline, 1)
                self.assertEqual(reward.context.num_deleted_lines_baseline, 1)

                self.assertEqual(reward.context.num_added_lines_generated, 1)
                self.assertEqual(reward.context.num_deleted_lines_generated, 1)

                self.assertEqual(reward.context.num_common_lines_baseline, 11)
                self.assertEqual(reward.context.num_common_lines_generated, 11)

                self.assertAlmostEqual(reward.context.test_pass, 1.0, places=2)
                self.assertAlmostEqual(reward.context.similarity_baseline, (11 / (11 + 1 + 1)), places=2)
                self.assertAlmostEqual(reward.context.similarity_generated, (11 / (11 + 1 + 1)), places=2)

                self.assertAlmostEqual(reward.test_reward, 1.0, places=2)
                self.assertAlmostEqual(reward.similarity_reward, 1.0, places=2)
                self.assertAlmostEqual(reward.total_reward, 1.0, places=2)

        async def test_reward_function_equal_difference_wrong_solution(self):
                buggy_code = '''
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
                correct_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                
                generated_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                mock_problem_manager = AsyncMock()
                mock_problem_manager.evaluate_submission.return_value = 0.99

                method = LLMMethod("vendor1", "model1", mock_problem_manager)

                buggy_code = clean_source_code(buggy_code)
                correct_code = clean_source_code(correct_code)

                reward = await method.compute_reward(problem_id = "problem1", 
                                        buggy_code=buggy_code,
                                        generated_code=generated_code, 
                                        correct_code=correct_code)
                
                self.assertEqual(reward.context.num_lines_buggy_solution, 12)
                self.assertEqual(reward.context.num_lines_correct_solution, 12)
                self.assertEqual(reward.context.num_lines_generated_solution, 12)

                self.assertEqual(reward.context.num_added_lines_baseline, 1)
                self.assertEqual(reward.context.num_deleted_lines_baseline, 1)

                self.assertEqual(reward.context.num_added_lines_generated, 1)
                self.assertEqual(reward.context.num_deleted_lines_generated, 1)

                self.assertEqual(reward.context.num_common_lines_baseline, 11)
                self.assertEqual(reward.context.num_common_lines_generated, 11)

                self.assertAlmostEqual(reward.context.test_pass, 0.99, places=2)
                self.assertAlmostEqual(reward.context.similarity_baseline, (11 / (11 + 1 + 1)), places=2)
                self.assertAlmostEqual(reward.context.similarity_generated, (11 / (11 + 1 + 1)), places=2)

                self.assertAlmostEqual(reward.test_reward, 0, places=2)
                self.assertAlmostEqual(reward.similarity_reward, 1.0, places=2)
                self.assertAlmostEqual(reward.total_reward, 0, places=2)

        async def test_reward_function_from_scratch(self):
                buggy_code = '''
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
                correct_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                
                generated_code = '''
                                #include<bits/stdc++.h>
                                void main() {
                                        int s = 10 * 11 / 2
                                std::cout << s << std::endl;}
                                '''
                mock_problem_manager = AsyncMock()
                mock_problem_manager.evaluate_submission.return_value = 1

                method = LLMMethod("vendor1", "model1", mock_problem_manager)

                buggy_code = clean_source_code(buggy_code)
                correct_code = clean_source_code(correct_code)

                reward = await method.compute_reward(problem_id = "problem1", 
                                        buggy_code=buggy_code,
                                        generated_code=generated_code, 
                                        correct_code=correct_code)
                
                self.assertEqual(reward.context.num_lines_buggy_solution, 12)
                self.assertEqual(reward.context.num_lines_correct_solution, 12)
                self.assertEqual(reward.context.num_lines_generated_solution, 4)

                self.assertEqual(reward.context.num_added_lines_baseline, 1)
                self.assertEqual(reward.context.num_deleted_lines_baseline, 1)

                self.assertEqual(reward.context.num_added_lines_generated, 4)
                self.assertEqual(reward.context.num_deleted_lines_generated, 12)

                self.assertEqual(reward.context.num_common_lines_baseline, 11)
                self.assertEqual(reward.context.num_common_lines_generated, 0)

                self.assertAlmostEqual(reward.context.test_pass, 1, places=2)
                self.assertAlmostEqual(reward.context.similarity_baseline, (11 / (11 + 1 + 1)), places=2)
                self.assertAlmostEqual(reward.context.similarity_generated, 0, places=2)

                self.assertAlmostEqual(reward.test_reward, 1, places=2)
                self.assertAlmostEqual(reward.similarity_reward, 0, places=2)
                self.assertAlmostEqual(reward.total_reward, 0, places=2)

        async def test_reward_function_better_than_baseline(self):
                buggy_code = '''
                                        #include<iostream>
                                using namespace std;
                                int main() {
                                        int s = 0;
                                        for(int i=1;i<10;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                correct_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                
                generated_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        for(int i=1;i<=10;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                mock_problem_manager = AsyncMock()
                mock_problem_manager.evaluate_submission.return_value = 1

                method = LLMMethod("vendor1", "model1", mock_problem_manager)

                buggy_code = clean_source_code(buggy_code)
                correct_code = clean_source_code(correct_code)

                reward = await method.compute_reward(problem_id = "problem1", 
                                        buggy_code=buggy_code,
                                        generated_code=generated_code, 
                                        correct_code=correct_code)
                
                self.assertEqual(reward.context.num_lines_buggy_solution, 11)
                self.assertEqual(reward.context.num_lines_correct_solution, 12)
                self.assertEqual(reward.context.num_lines_generated_solution, 11)

                self.assertEqual(reward.context.num_added_lines_baseline, 2)
                self.assertEqual(reward.context.num_deleted_lines_baseline, 1)

                self.assertEqual(reward.context.num_added_lines_generated, 1)
                self.assertEqual(reward.context.num_deleted_lines_generated, 1)

                self.assertEqual(reward.context.num_common_lines_baseline, 10)
                self.assertEqual(reward.context.num_common_lines_generated, 10)

                self.assertAlmostEqual(reward.context.test_pass, 1, places=2)
                similarity_baseline = 10 / (10 + 2 + 1)
                similarity_generated = 10 / (10 + 1 + 1)

                self.assertAlmostEqual(reward.context.similarity_baseline, similarity_baseline, places=2)
                self.assertAlmostEqual(reward.context.similarity_generated, similarity_generated, places=2)

                self.assertAlmostEqual(reward.test_reward, 1, places=2)
                self.assertAlmostEqual(reward.similarity_reward, 1, places=2)
                self.assertAlmostEqual(reward.total_reward, 1, places=2)
        
        async def test_reward_function_worst_than_baseline(self):
                buggy_code = '''
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
                correct_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                
                generated_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        s = n * (n + 1) / 2;
                                cout << s << endl;
                                return 0;
                                }
                                '''
                mock_problem_manager = AsyncMock()
                mock_problem_manager.evaluate_submission.return_value = 1

                method = LLMMethod("vendor1", "model1", mock_problem_manager)

                buggy_code = clean_source_code(buggy_code)
                correct_code = clean_source_code(correct_code)

                reward = await method.compute_reward(problem_id = "problem1", 
                                        buggy_code=buggy_code,
                                        generated_code=generated_code, 
                                        correct_code=correct_code)
                
                self.assertEqual(reward.context.num_lines_buggy_solution, 12)
                self.assertEqual(reward.context.num_lines_correct_solution, 12)
                self.assertEqual(reward.context.num_lines_generated_solution, 9)

                self.assertEqual(reward.context.num_added_lines_baseline, 1)
                self.assertEqual(reward.context.num_deleted_lines_baseline, 1)

                self.assertEqual(reward.context.num_added_lines_generated, 1)
                self.assertEqual(reward.context.num_deleted_lines_generated, 4)

                self.assertEqual(reward.context.num_common_lines_baseline, 11)
                self.assertEqual(reward.context.num_common_lines_generated, 8)

                self.assertAlmostEqual(reward.context.test_pass, 1, places=2)
                similarity_baseline = 11 / (11 + 1 + 1)
                similarity_generated = 8 / (8 + 1 + 4)

                self.assertAlmostEqual(reward.context.similarity_baseline, similarity_baseline, places=2)
                self.assertAlmostEqual(reward.context.similarity_generated, similarity_generated, places=2)

                self.assertAlmostEqual(reward.test_reward, 1, places=2)
                self.assertAlmostEqual(reward.similarity_reward, similarity_generated / similarity_baseline , places=2)
                self.assertAlmostEqual(reward.total_reward, similarity_generated / similarity_baseline, places=2)
                
        async def test_reward_function_extra_work(self):
                buggy_code = '''
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
                correct_code = '''
                                #include<iostream>
                                using namespace std; 
                                int main() {
                                        int s = 0;
                                        for(int i=1;i<=10;++i)
                                        {
                                                s += i;
                                        }
                                cout << s << endl;
                                return 0;
                                }
                                '''
                
                generated_code = '''
                                #include<iostream>
                                using namespace std; 
                                int sum_first_n_numbers(int n) {
                                        int s = 0;
                                        for(int i=1;i<=n;++i)
                                        {
                                                s += i;
                                        }
                                        return s;
                                }
                                int main() {
                                        int s = 0;
                                        int n = 10;
                                        s = sum_first_n_numbers(n);
                                cout << s << endl;
                                return 0;
                                }
                                '''
                mock_problem_manager = AsyncMock()
                mock_problem_manager.evaluate_submission.return_value = 1.0

                method = LLMMethod("vendor1", "model1", mock_problem_manager)

                buggy_code = clean_source_code(buggy_code)
                correct_code = clean_source_code(correct_code)

                reward = await method.compute_reward(problem_id = "problem1", 
                                        buggy_code=buggy_code,
                                        generated_code=generated_code, 
                                        correct_code=correct_code)
                
                self.assertEqual(reward.context.num_lines_buggy_solution, 12)
                self.assertEqual(reward.context.num_lines_correct_solution, 11)
                self.assertEqual(reward.context.num_lines_generated_solution, 17)

                self.assertEqual(reward.context.num_added_lines_baseline, 1)
                self.assertEqual(reward.context.num_deleted_lines_baseline, 2)

                self.assertEqual(reward.context.num_added_lines_generated, 9)
                self.assertEqual(reward.context.num_deleted_lines_generated, 4)

                self.assertEqual(reward.context.num_common_lines_baseline, 10)
                self.assertEqual(reward.context.num_common_lines_generated, 8)

                similarity_baseline = (10 / (10 + 1 + 2))
                similarity_generated = (8 / (8 + 9 + 4))

                self.assertAlmostEqual(reward.context.test_pass, 1.0, places=2)
                self.assertAlmostEqual(reward.context.similarity_baseline, similarity_baseline, places=2)
                self.assertAlmostEqual(reward.context.similarity_generated, similarity_generated, places=2)

                self.assertAlmostEqual(reward.test_reward, 1.0, places=2)
                self.assertAlmostEqual(reward.similarity_reward, similarity_generated / similarity_baseline, places=2)
                self.assertAlmostEqual(reward.total_reward, similarity_generated / similarity_baseline, places=2)

        
        
        
        