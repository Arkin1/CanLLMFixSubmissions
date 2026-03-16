import unittest
from utils import clean_source_code
class TestCleanCode(unittest.TestCase):
    def test_code_parser_no_cpp(self):
        code = '''
                    
        #include <iostream>
        #include <vector>
        #include <algorithm>

        /**
        * Problem Analysis:
        * We have rectangles defined by (x_i, y_i) and a cost a_i.
        * We want to maximize (Area(Union) - Sum(a_i)).
        * The condition \"no nested rectangles\" means for any two rectangles i and j,
        * if we sort them by x-coordinate, then their y-coordinates must be sorted in reverse order.
        * Let's sort the rectangles such that x_1 < x_2 < ... < x_n.
        * Because there are no nested rectangles, it must be true that y_1 > y_2 > ... > y_n.
        * 
        * The union of these rectangles forms a \"staircase\" shape.
        * Let the selected rectangles be indices i_1, i_2, ..., i_k (sorted by x).
        * The area of the union is sum of areas of individual rectangles minus the overlapping parts,
        * but since they are nested in the \"staircase\" fashion (x increases, y decreases),
        * the union area is simply:
        * x_{i_1} * y_{i_1} + (x_{i_2} - x_{i_1}) * y_{i_2} + ... + (x_{i_k} - x_{i_{k-1}}) * y_{i_k}
        * Actually, it's easier to think about this as DP.
        * Let the rectangles be sorted such that x_1 < x_2 < ... < x_n.
        * Then y_1 > y_2 > ... > y_n.
        * If we pick a set of indices, the contribution to the total area is:
        * (x_{i_1} * y_{i_1}) + (x_{i_2} - x_{i_1}) * y_{i_2} + ...
        * This is equivalent to:
        * x_{i_1}(y_{i_1} - y_{i_2}) + x_{i_2}(y_{i_2} - y_{i_3}) + ... + x_{i_k} * y_{i_k}
        * Wait, let's re-evaluate the objective function:
        * Objective = Area(Union) - Sum(a_i)
        * Let dp[i] be the max value we can get using a subset of first i rectangles where the i-th rectangle is included.
        * To calculate dp[i], we can transition from any j < i (or start a new sequence).
        * The gain of adding rectangle i after rectangle j is:
        * (Area_union_with_i_and_j) - (Area_union_with_j) - a_i
        * The area added by rectangle i when rectangle j is already present (where x_j < x_i and y_j > y_i):
        * Since x_j < x_i and y_j > y_i, rectangle i extends the width from x_j to x_i but only for the height y_i.
        * Area added = (x_i - x_j) * y_i.
        * So, dp[i] = max(x_i * y_i - a_i, max_{j < i} (dp[j] + (x_i - x_j) * y_i - a_i))
        * dp[i] = max(x_i * y_i - a_i, max_{j < i} (dp[j] - x_j * y_i) + x_i * y_i - a_i)
        * 
        * This is a standard Convex Hull Trick (CHT) form: y = mx + c.
        * Here:
        * dp[j] - x_j * y_i is the term to maximize.
        * m = -x_j
        * x = y_i
        * c = dp[j]
        * Since we want to maximize, and y_i is decreasing (as we process from 1 to n),
        * we need a CHT structure that handles query points. 
        * Since y_i is decreasing, the query points are monotonic.
        * Slopes (-x_j) are also monotonic (since x_j is increasing).
        * Thus, we can use a deque or a simple linear scan with a stack.
        */

        using namespace std;

        struct Rectangle {
            long long x, y, a;
        };

        // Line structure for CHT: y = mx + c
        struct Line {
            long long m, c;
            long long eval(long long x) { return m * x + c; }
        };

        int main() {
            ios_base::sync_with_stdio(false);
            cin.tie(NULL);

            int n;
            cin >> n;
            vector<Rectangle> rects(n);
            for (int i = 0; i < n; ++i) {
                cin >> rects[i].x >> rects[i].y >> rects[i].a;
            }

            // Sort by x increasing. Because no nested rectangles, y must be decreasing.
            sort(rects.begin(), rects.end(), [](const Rectangle& a, const Rectangle& b) {
                return a.x < b.x;
            });

            // dp[i] = max value ending at rectangle i
            // dp[i] = x_i * y_i - a_i + max(0, max_{j < i} (dp[j] - x_j * y_i))
            // We want to maintain lines of form y = -x_j * Y + dp[j].
            // Since x_j is increasing, slopes -x_j are decreasing.
            // Query points Y = y_i are decreasing.
            // Standard CHT for maximizing.
            
            vector<Line> hull;
            auto add_line = [&](long long m, long long c) {
                // Since slopes are decreasing, we can just maintain the lower convex hull logic for max
                // Standard condition for lines l1, l2, l3 (m1 > m2 > m3):
                // Intersect(l1, l2) < Intersect(l2, l3)
                // (c2 - c1) / (m1 - m2) < (c3 - c2) / (m2 - m3)
                while (hull.size() >= 2) {
                    Line l2 = hull.back();
                    Line l1 = hull[hull.size() - 2];
                    // Check intersection intersection of (l1, l2) vs (l2, new)
                    // (l2.c - l1.c) * (l2.m - m) >= (c - l2.c) * (l1.m - l2.m)
                    // Be careful with overflows. Use __int128 if necessary.
                    if ((__int128)(l2.c - l1.c) * (l2.m - m) >= (__int128)(c - l2.c) * (l1.m - l2.m)) {
                        hull.pop_back();
                    } else {
                        break;
                    }
                }
                hull.push_back({m, c});
            };

            auto query = [&](long long x) {
                if (hull.empty()) return -2e18; // Should not happen given logic
                // Since query points x are decreasing, the optimal line moves to the left/right.
                // Actually, for maximizing, we can just pop from the front/back if queries are monotonic.
                // Queries y_i are decreasing. Slopes -x_j are decreasing.
                // This is standard monotonic CHT.
                while (hull.size() >= 2 && hull[0].eval(x) <= hull[1].eval(x)) {
                    hull.erase(hull.begin());
                }
                return hull[0].eval(x);
            };

            long long ans = 0;
            // Initial DP value
            // dp[0] = rects[0].x * rects[0].y - rects[0].a
            // But we should consider adding lines after processing.
            
            for (int i = 0; i < n; ++i) {
                long long current_val = (long long)rects[i].x * rects[i].y - rects[i].a;
                if (!hull.empty()) {
                    current_val = max(current_val, query(rects[i].y) + (long long)rects[i].x * rects[i].y - rects[i].a);
                }
                ans = max(ans, current_val);
                add_line(-rects[i].x, current_val);
            }

            cout << ans << endl;

            return 0;
        }'''
            
        correct_code = '''        #include <iostream>
        #include <vector>
        #include <algorithm>
        using namespace std;
        struct Rectangle {
            long long x, y, a;
        };
        struct Line {
            long long m, c;
            long long eval(long long x) { return m * x + c; }
        };
        int main() {
            ios_base::sync_with_stdio(false);
            cin.tie(NULL);
            int n;
            cin >> n;
            vector<Rectangle> rects(n);
            for (int i = 0; i < n; ++i) {
                cin >> rects[i].x >> rects[i].y >> rects[i].a;
            }
            sort(rects.begin(), rects.end(), [](const Rectangle& a, const Rectangle& b) {
                return a.x < b.x;
            });
            vector<Line> hull;
            auto add_line = [&](long long m, long long c) {
                while (hull.size() >= 2) {
                    Line l2 = hull.back();
                    Line l1 = hull[hull.size() - 2];
                    if ((__int128)(l2.c - l1.c) * (l2.m - m) >= (__int128)(c - l2.c) * (l1.m - l2.m)) {
                        hull.pop_back();
                    } else {
                        break;
                    }
                }
                hull.push_back({m, c});
            };
            auto query = [&](long long x) {
                if (hull.empty()) return -2e18; 
                while (hull.size() >= 2 && hull[0].eval(x) <= hull[1].eval(x)) {
                    hull.erase(hull.begin());
                }
                return hull[0].eval(x);
            };
            long long ans = 0;
            for (int i = 0; i < n; ++i) {
                long long current_val = (long long)rects[i].x * rects[i].y - rects[i].a;
                if (!hull.empty()) {
                    current_val = max(current_val, query(rects[i].y) + (long long)rects[i].x * rects[i].y - rects[i].a);
                }
                ans = max(ans, current_val);
                add_line(-rects[i].x, current_val);
            }
            cout << ans << endl;
            return 0;
        }'''
        clean_code = clean_source_code(code)

        self.assertEqual(clean_code, correct_code)