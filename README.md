# Most_Imp_qus

\
**Arrays:**
1. Two Sum - LeetCode: https://leetcode.com/problems/two-sum/
```cpp
// Pseudo-code for Two Sum
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> mp; // Create a map to store the numbers and their indices
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i]; // Calculate the complement required to achieve the target sum
        if (mp.count(complement)) { // If the complement is found in the map, return the indices
            return {mp[complement], i};
        }
        mp[nums[i]] = i; // Store the current number and its index in the map
    }
    return {}; // If no such pair found, return an empty vector
}
```

2. Best Time to Buy and Sell Stock - LeetCode: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
```cpp
// Pseudo-code for Best Time to Buy and Sell Stock
int maxProfit(vector<int>& prices) {
    int minPrice = INT_MAX; // Initialize minimum price to maximum possible value
    int maxProfit = 0; // Initialize maximum profit to 0
    for (int price : prices) {
        minPrice = min(minPrice, price); // Keep track of the minimum price encountered so far
        maxProfit = max(maxProfit, price - minPrice); // Calculate the maximum profit possible
    }
    return maxProfit;
}
```

3. Rotate Array - LeetCode: https://leetcode.com/problems/rotate-array/
```cpp
// Pseudo-code for Rotate Array
void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    k = k % n; // Handle cases where k is greater than the array size
    reverse(nums.begin(), nums.end()); // Reverse the entire array
    reverse(nums.begin(), nums.begin() + k); // Reverse the first 'k' elements
    reverse(nums.begin() + k, nums.end()); // Reverse the remaining elements
}
```

4. Product of Array Except Self - LeetCode: https://leetcode.com/problems/product-of-array-except-self/
```cpp
// Pseudo-code for Product of Array Except Self
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> leftProduct(n, 1), rightProduct(n, 1), result(n, 1);
    
    // Calculate product of elements to the left of each element
    for (int i = 1; i < n; i++) {
        leftProduct[i] = leftProduct[i - 1] * nums[i - 1];
    }
    
    // Calculate product of elements to the right of each element
    for (int i = n - 2; i >= 0; i--) {
        rightProduct[i] = rightProduct[i + 1] * nums[i + 1];
    }
    
    // Calculate the final product array by multiplying left and right products
    for (int i = 0; i < n; i++) {
        result[i] = leftProduct[i] * rightProduct[i];
    }
    
    return result;
}
```

5. Merge Intervals - LeetCode: https://leetcode.com/problems/merge-intervals/
```cpp
// Pseudo-code for Merge Intervals
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end()); // Sort intervals based on the starting point
    vector<vector<int>> mergedIntervals;
    for (auto interval : intervals) {
        if (mergedIntervals.empty() || interval[0] > mergedIntervals.back()[1]) {
            // If the mergedIntervals is empty or no overlap, add new interval to the result
            mergedIntervals.push_back(interval);
        } else {
            // If there is overlap, merge the intervals by updating the end point
            mergedIntervals.back()[1] = max(mergedIntervals.back()[1], interval[1]);
        }
    }
    return mergedIntervals;
}
```

6. Find First and Last Position of Element in Sorted Array - LeetCode: https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
```cpp
// Pseudo-code for Find First and Last Position of Element in Sorted Array
vector<int> searchRange(vector<int>& nums, int target) {
    int left = lower_bound(nums.begin(), nums.end(), target) - nums.begin();
    int right = upper_bound(nums.begin(), nums.end(), target) - nums.begin() - 1;
    
    if (left <= right) {
        return {left, right};
    } else {
        return {-1, -1};
    }
}
```


Certainly! Here are the pseudo-code/algo in C++ for the remaining questions along with short comments and links to their LeetCode pages:

**Strings:**
1. Reverse String - LeetCode: https://leetcode.com/problems/reverse-string/
```cpp
// Pseudo-code for Reverse String
void reverseString(vector<char>& s) {
    int left = 0, right = s.size() - 1;
    while (left < right) {
        swap(s[left], s[right]); // Swap characters from both ends to reverse the string
        left++;
        right--;
    }
}
```

2. Valid Palindrome - LeetCode: https://leetcode.com/problems/valid-palindrome/
```cpp
// Pseudo-code for Valid Palindrome
bool isPalindrome(string s) {
    int left = 0, right = s.size() - 1;
    while (left < right) {
        while (left < right && !isalnum(s[left])) left++; // Skip non-alphanumeric characters from the left
        while (left < right && !isalnum(s[right])) right--; // Skip non-alphanumeric characters from the right
        if (tolower(s[left]) != tolower(s[right])) return false; // Check if characters are equal ignoring case
        left++;
        right--;
    }
    return true;
}
```

3. Longest Substring Without Repeating Characters - LeetCode: https://leetcode.com/problems/longest-substring-without-repeating-characters/
```cpp
// Pseudo-code for Longest Substring Without Repeating Characters
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> charMap; // Map to store characters and their indices
    int maxLength = 0, left = 0;
    for (int right = 0; right < s.size(); right++) {
        if (charMap.count(s[right])) {
            left = max(left, charMap[s[right]] + 1); // Update left pointer to avoid duplicates
        }
        charMap[s[right]] = right; // Update the index of the current character
        maxLength = max(maxLength, right - left + 1); // Update the maximum length
    }
    return maxLength;
}
```

4. Group Anagrams - LeetCode: https://leetcode.com/problems/group-anagrams/
```cpp
// Pseudo-code for Group Anagrams
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> anagramGroups; // Map to store anagram groups
    for (string str : strs) {
        string sortedStr = str;
        sort(sortedStr.begin(), sortedStr.end()); // Sort each string to find its anagram group
        anagramGroups[sortedStr].push_back(str); // Add the current string to its anagram group
    }
    vector<vector<string>> result;
    for (auto& group : anagramGroups) {
        result.push_back(group.second); // Add all anagram groups to the result
    }
    return result;
}
```

5. Longest Palindromic Substring - LeetCode: https://leetcode.com/problems/longest-palindromic-substring/
```cpp
// Pseudo
Certainly! Here's the continuation of the pseudo-code for the "Longest Palindromic Substring" problem:

```cpp
        int len = max(len1, len2);
        
        if (len > maxLength) {
            // Update the start index and maximum length if a longer palindrome is found
            start = i - (len - 1) / 2;
            maxLength = len;
        }
    }
    
    // Extract and return the longest palindromic substring
    return s.substr(start, maxLength);
}

// Helper function to expand around the center to find the length of the palindrome
int expandAroundCenter(const string& s, int left, int right) {
    while (left >= 0 && right < s.size() && s[left] == s[right]) {
        left--;
        right++;
    }
    // Return the length of the palindrome (right - left - 1)
    return right - left - 1;
}
```

**Linked Lists:**
1. Reverse Linked List - LeetCode: https://leetcode.com/problems/reverse-linked-list/
```cpp
// Pseudo-code for Reverse Linked List
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    while (head) {
        ListNode* nextNode = head->next; // Save the next node
        head->next = prev; // Reverse the pointer direction
        prev = head; // Move prev pointer to the current head
        head = nextNode; // Move head to the next node
    }
    return prev; // Return the new head (prev points to the last node)
}
```

2. Merge Two Sorted Lists - LeetCode: https://leetcode.com/problems/merge-two-sorted-lists/
```cpp
// Pseudo-code for Merge Two Sorted Lists
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy(0); // Create a dummy node to simplify the code
    ListNode* current = &dummy;
    
    while (l1 && l2) {
        if (l1->val <= l2->val) {
            current->next = l1;
            l1 = l1->next;
        } else {
            current->next = l2;
            l2 = l2->next;
        }
        current = current->next;
    }
    
    current->next = l1 ? l1 : l2; // Attach the remaining list
    
    return dummy.next;
}
```

3. Linked List Cycle - LeetCode: https://leetcode.com/problems/linked-list-cycle/
```cpp
// Pseudo-code for Linked List Cycle
bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            return true; // If there's a cycle, slow and fast pointers will meet
        }
    }
    return false;
}
```

4. Remove Nth Node From End of List - LeetCode: https://leetcode.com/problems/remove-nth-node-from-end-of-list/
```cpp
// Pseudo-code for Remove Nth Node From End of List
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0); // Create a dummy node to handle edge cases
    dummy.next = head;
    ListNode* slow = &dummy;
    ListNode* fast = &dummy;
    
    // Move the fast pointer n+1 steps ahead
    for (int i = 0; i <= n; i++) {
        fast = fast->next;
    }
    
    // Move both pointers until the fast reaches the end
    while (fast) {
        slow = slow->next;
        fast = fast->next;
    }
    
    // Remove the Nth node from the end
    ListNode* temp = slow->next;
    slow->next = slow->next->next;
    delete temp;
    
    return dummy.next;
}
```

5. Copy List with Random Pointer - LeetCode: https://leetcode.com/problems/copy-list-with-random-pointer/
```cpp
// Pseudo-code for Copy List with Random Pointer
Node* copyRandomList(Node* head) {
    if (!head) return nullptr;
    
    unordered_map<Node*, Node*> nodeMap; // Map to store original nodes and their copies
    
    Node* current = head;
    while (current) {
        nodeMap[current] = new Node(current->val); // Create a copy of the current node
        current = current->next;
    }
    
    current = head;
    while (current) {
        nodeMap[current]->next = nodeMap[current->next]; // Link the next pointers
        nodeMap[current]->random = nodeMap[current->random]; // Link the random pointers
        current = current->next;
    }
    
    return nodeMap[head]; // Return the head of the copied linked list
}
```

6. Add Two Numbers - LeetCode: https://leetcode.com/problems/add-two-numbers/
```cpp
// Pseudo-code for Add Two Numbers
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode dummy(0); // Create a dummy node to simplify the code
    ListNode* current = &dummy;
    int carry = 0;
    
    while (l1 || l2 || carry) {
        int sum = carry;
        if (l1) {
            sum += l1->val;
            l1 = l1->next;
        }
        if (l2) {
            sum += l2->val;
            l2 = l2->next;
        }
        
        carry = sum / 10;
        current->next = new ListNode(sum % 10);
        current = current->next;
    }
    
    return dummy.next;
}
```


Certainly! Here's the continuation of the pseudo-code for the remaining questions along with short comments and links to their LeetCode pages:

**Trees:**
1. Maximum Depth of Binary Tree - LeetCode: https://leetcode.com/problems/maximum-depth-of-binary-tree/
```cpp
// Pseudo-code for Maximum Depth of Binary Tree
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

2. Validate Binary Search Tree - LeetCode: https://leetcode.com/problems/validate-binary-search-tree/
```cpp
// Pseudo-code for Validate Binary Search Tree
bool isValidBST(TreeNode* root, TreeNode* minNode = nullptr, TreeNode* maxNode = nullptr) {
    if (!root) return true;
    
    if ((minNode && root->val <= minNode->val) || (maxNode && root->val >= maxNode->val)) {
        return false;
    }
    
    return isValidBST(root->left, minNode, root) && isValidBST(root->right, root, maxNode);
}
```

3. Binary Tree Level Order Traversal - LeetCode: https://leetcode.com/problems/binary-tree-level-order-traversal/
```cpp
// Pseudo-code for Binary Tree Level Order Traversal
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            currentLevel.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        result.push_back(currentLevel);
    }
    
    return result;
}
```

4. Lowest Common Ancestor of a Binary Tree - LeetCode: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
```cpp
// Pseudo-code for Lowest Common Ancestor of a Binary Tree
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    
    if (left && right) return root; // If both p and q are found in the left and right subtrees
    return left ? left : right; // Return the non-null value among left and right
}
```

5. Symmetric Tree - LeetCode: https://leetcode.com/problems/symmetric-tree/
```cpp
// Pseudo-code for Symmetric Tree
bool isSymmetric(TreeNode* root) {
    if (!root) return true;
    return isMirror(root->left, root->right);
}

bool isMirror(TreeNode* leftSubtree, TreeNode* rightSubtree) {
    if (!leftSubtree && !rightSubtree) return true;
    if (!leftSubtree || !rightSubtree) return false;
    return (leftSubtree->val == rightSubtree->val) &&
           isMirror(leftSubtree->left, rightSubtree->right) &&
           isMirror(leftSubtree->right, rightSubtree->left);
}
```

6. Serialize and Deserialize Binary Tree - LeetCode: https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
```cpp
// Pseudo-code for Serialize and Deserialize Binary Tree
string serialize(TreeNode* root) {
    if (!root) return "X"; // Represent null nodes with "X"
    return to_string(root->val) + "," + serialize(root->left) + "," + serialize(root->right);
}

TreeNode* deserialize(string data) {
    stringstream ss(data);
    return deserializeHelper(ss);
}

TreeNode* deserializeHelper(stringstream& ss) {
    string val;
    getline(ss, val, ',');
    if (val == "X") return nullptr;
    
    TreeNode* root = new TreeNode(stoi(val));
    root->left = deserializeHelper(ss);
    root->right = deserializeHelper(ss);
    return root;
}
```

(Note: Please check the provided LeetCode links for detailed problem descriptions and solutions.)

**Dynamic Programming:**
1. Climbing Stairs - LeetCode: https://leetcode.com/problems/climbing-stairs/
```cpp
// Pseudo-code for Climbing Stairs
int climbStairs(int n) {
    if (n <= 2) return n;
    int dp[n + 1];
    dp[0] = 0;
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2]; // Fibonacci-like sequence
    }
    return dp[n];
}
```

2. Longest Increasing Subsequence - LeetCode: https://leetcode.com/problems/longest-increasing-subsequence/
```cpp
// Pseudo-code for Longest Increasing Subsequence
int lengthOfLIS(vector<int>& nums) {
    vector<int> dp(nums.size(), 1); // Initialize the dp array with 1
    int maxLength = 1;
    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = max(dp[i], dp[j] + 1); // Update LIS length for the current index
                maxLength = max(maxLength, dp[i]); // Update the overall maximum length
            }
        }
    }
    return maxLength;
}
```

3. Coin Change - LeetCode: https://leetcode.com/problems/coin-change/
```cpp
// Pseudo-code for Coin Change
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1); // Initialize dp array with a value greater than amount
    dp[0] = 0; // Base case: 0 coins needed to make amount 0
    
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] = min(dp[i], dp[i - coin] + 1); // Update minimum coins needed for each amount
        }
    }
    
    return dp[amount] > amount ? -1 : dp[amount];
}
```

4. Maximum Subarray - LeetCode: https://leetcode.com/problems/maximum-subarray/
```cpp
// Pseudo-code for Maximum Subarray
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0], currentSum = nums[0];
    for (int i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]); // Choose whether to start a new subarray or extend the current one
        maxSum = max(maxSum, currentSum); // Update the maximum sum seen

 so far
    }
    return maxSum;
}
```

5. Word Break - LeetCode: https://leetcode.com/problems/word-break/
```cpp
// Pseudo-code for Word Break
bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> wordSet(wordDict.begin(), wordDict.end()); // Convert wordDict to a set for faster lookup
    vector<bool> dp(s.size() + 1, false); // Initialize dp array with false
    dp[0] = true; // Empty string is always a valid word
    
    for (int i = 1; i <= s.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && wordSet.count(s.substr(j, i - j))) {
                dp[i] = true; // Update dp[i] if a valid word is found
                break;
            }
        }
    }
    
    return dp[s.size()]; // Return whether the entire string can be segmented
}
```

6. Unique Paths - LeetCode: https://leetcode.com/problems/unique-paths/
```cpp
// Pseudo-code for Unique Paths
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1)); // Initialize dp array with 1
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]; // Calculate the unique paths for each cell
        }
    }
    return dp[m - 1][n - 1]; // Return the number of unique paths to the destination
}
```

(Note: Please check the provided LeetCode links for detailed problem descriptions and solutions.)

**Sorting and Searching:**
1. Merge Sort:
```cpp
// Pseudo-code for Merge Sort
void mergeSort(vector<int>& nums, int left, int right) {
    if (left >= right) return;
    
    int mid = left + (right - left) / 2;
    mergeSort(nums, left, mid);
    mergeSort(nums, mid + 1, right);
    merge(nums, left, mid, right);
}

void merge(vector<int>& nums, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid && j <= right) {
        if (nums[i] <= nums[j]) {
            temp[k++] = nums[i++];
        } else {
            temp[k++] = nums[j++];
        }
    }
    
    while (i <= mid) temp[k++] = nums[i++];
    while (j <= right) temp[k++] = nums[j++];
    
    for (int p = 0; p < temp.size(); p++) {
        nums[left + p] = temp[p];
    }
}
```

2. Quick Sort:
```cpp
// Pseudo-code for Quick Sort
void quickSort(vector<int>& nums, int left, int right) {
    if (left >= right) return;
    
    int pivotIndex = partition(nums, left, right);
    quickSort(nums, left, pivotIndex - 1);
    quickSort(nums, pivotIndex + 1, right);
}

int partition(vector<int>& nums, int left, int right) {
    int pivot = nums[right]; // Choose the rightmost element as the pivot
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (nums[j] <= pivot) {
            i++;
            swap(nums[i], nums[j]); // Move smaller elements to the left
        }
    }
    
    swap(nums[i + 1], nums[right]); // Place the pivot at its correct position
    return i + 1;
}
```

3. Binary Search - LeetCode: https://leetcode.com/problems/binary-search/
```cpp
// Pseudo-code for Binary Search
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

4. First Bad Version - LeetCode: https://leetcode.com/problems/first-bad-version/
```cpp
// Pseudo-code for First Bad Version
int firstBadVersion(int n) {
    int left = 1, right = n;
    while (left < right) {
        int mid = left + (right - left) / 2

;
        if (isBadVersion(mid)) right = mid;
        else left = mid + 1;
    }
    return left;
}
```

5. Search in Rotated Sorted Array - LeetCode: https://leetcode.com/problems/search-in-rotated-sorted-array/
```cpp
// Pseudo-code for Search in Rotated Sorted Array
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) right = mid - 1;
            else left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right]) left = mid + 1;
            else right = mid - 1;
        }
    }
    return -1;
}
```

6. Kth Largest Element in an Array - LeetCode: https://leetcode.com/problems/kth-largest-element-in-an-array/
```cpp
// Pseudo-code for Kth Largest Element in an Array
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (int num : nums) {
        minHeap.push(num); // Keep the size of the heap to k
        if (minHeap.size() > k) {
            minHeap.pop();
        }
    }
    
    return minHeap.top(); // The top element of the minHeap is the kth largest
}
```

(Note: Please check the provided LeetCode links for detailed problem descriptions and solutions.)

**Graphs:**
1. DFS and BFS Traversal:
DFS (Depth-First Search):
```cpp
// Pseudo-code for DFS Traversal
void dfs(Node* node, unordered_set<Node*>& visited) {
    if (!node || visited.count(node)) return;
    visited.insert(node);
    for (Node* neighbor : node->neighbors) {
        dfs(neighbor, visited);
    }
}
```

BFS (Breadth-First Search):
```cpp
// Pseudo-code for BFS Traversal
void bfs(Node* start) {
    queue<Node*> q;
    unordered_set<Node*> visited;
    q.push(start);
    visited.insert(start);
    
    while (!q.empty()) {
        Node* node = q.front();
        q.pop();
        // Process the node as needed
        for (Node* neighbor : node->neighbors) {
            if (!visited.count(neighbor)) {
                q.push(neighbor);
                visited.insert(neighbor);
            }
        }
    }
}
```

2. Number of Islands - LeetCode: https://leetcode.com/problems/number-of-islands/
```cpp
// Pseudo-code for Number of Islands
int numIslands(vector<vector<char>>& grid) {
    int rows = grid.size();
    if (rows == 0) return 0;
    int cols = grid[0].size();
    int numIslands = 0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == '1') {
                numIslands++;
                dfsMarkIsland(grid, i, j);
            }
        }
    }
    
    return numIslands;
}

void dfsMarkIsland(vector<vector<char>>& grid, int row, int col) {
    int rows = grid.size();
    int cols = grid[0].size();
    if (row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] != '1') return;
    
    grid[row][col] = '0'; // Mark the current cell as visited
    // Visit the neighboring cells in all four directions
    dfsMarkIsland(grid, row + 1, col);
    dfsMarkIsland(grid, row - 1, col);
    dfsMarkIsland(grid, row, col + 1);
    dfsMarkIsland(grid, row, col - 1);
}
```

3. Course Schedule - LeetCode: https://leetcode.com/problems/course-schedule/
```cpp
// Pseudo-code for Course Schedule
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses, vector<int>());
    vector<int> indegree(numCourses, 0);
    
    // Build the graph and calculate the indegree for each node
    for (const auto& edge : prerequisites) {
        graph[edge[1]].push_back(edge[0]);
        indegree[edge[0]]++;
    }
    
    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree

[i] == 0) {
            q.push(i);
        }
    }
    
    int count = 0;
    while (!q.empty()) {
        int course = q.front();
        q.pop();
        count++;
        
        for (int nextCourse : graph[course]) {
            indegree[nextCourse]--;
            if (indegree[nextCourse] == 0) {
                q.push(nextCourse);
            }
        }
    }
    
    return count == numCourses;
}
```

4. Clone Graph - LeetCode: https://leetcode.com/problems/clone-graph/
```cpp
// Pseudo-code for Clone Graph
Node* cloneGraph(Node* node) {
    if (!node) return nullptr;
    
    unordered_map<Node*, Node*> nodeMap;
    return dfsClone(node, nodeMap);
}

Node* dfsClone(Node* node, unordered_map<Node*, Node*>& nodeMap) {
    if (!node || nodeMap.count(node)) return nodeMap[node];
    
    Node* newNode = new Node(node->val);
    nodeMap[node] = newNode;
    
    for (Node* neighbor : node->neighbors) {
        newNode->neighbors.push_back(dfsClone(neighbor, nodeMap));
    }
    
    return newNode;
}
```

5. Shortest Path in Binary Matrix - LeetCode: https://leetcode.com/problems/shortest-path-in-binary-matrix/
```cpp
// Pseudo-code for Shortest Path in Binary Matrix
int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
    int n = grid.size();
    if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) return -1;
    
    vector<vector<int>> directions = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    queue<pair<int, int>> q;
    q.push({0, 0});
    grid[0][0] = 1;
    
    int steps = 1;
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            int row = q.front().first;
            int col = q.front().second;
            q.pop();
            if (row == n - 1 && col == n - 1) return steps;
            for (const auto& dir : directions) {
                int newRow = row + dir[0];
                int newCol = col + dir[1];
                if (newRow >= 0 && newRow < n && newCol >= 0 && newCol < n && grid[newRow][newCol] == 0) {
                    q.push({newRow, newCol});
                    grid[newRow][newCol] = 1;
                }
            }
        }
        steps++;
    }
    return -1;
}
```

6. Network Delay Time - LeetCode: https://leetcode.com/problems/network-delay-time/
```cpp
// Pseudo-code for Network Delay Time
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<pair<int, int>>> graph(n + 1);
    for (const auto& time : times) {
        graph[time[0]].push_back({time[1], time[2]});
    }
    
    vector<int> dist(n + 1, INT_MAX);
    dist[k] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, k});
    
    while (!pq.empty()) {
        int currentDist = pq.top().first;
        int node = pq.top().second;
        pq.pop();
        
        if (currentDist > dist[node]) continue;
        
        for (const auto& neighbor : graph[node]) {
            int newDist = currentDist + neighbor.second;
            if (newDist < dist[neighbor.first]) {
                dist[neighbor.first] = newDist;
                pq.push({newDist, neighbor.first});
            }
        }
    }
    
    int maxDelay = *max_element(dist.begin() + 1, dist.end());
    return maxDelay == INT_MAX ? -1 : maxDelay;
}
```


Good luck with your preparation and placement drive! 
