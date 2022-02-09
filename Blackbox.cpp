#include <bits/stdc++.h>
using namespace std;
//Fact Blog

//except power of two every other number has atleast one odd devisor.
//number of 2021 needed to make x after using all 2020
//int cnt2021 = n % 2020;
//power of 2 = ceil(log2(7)));
//no matter how huge the power of 2 is (t <<= 1;t %= mod);
// x+mod to handle -ve val

//x*2>m alag hota hai x>((m+1)/2) try both if one fails

//log2(x) when used with 576460752303423487 should give
//58 but gives 59 dont use when 64 required

int add(int a, int b) {
	return (a + b) % mod;
}

int mul(int a, int b) {
	return ((a % mod) * (b % mod)) % mod;
}


int kmp(string a, int l) {
	// string temp = patt + "#" + a;
	int arr[a.length()];
	arr[0] = 0;
	int i = 1, len = 0;
	while (i < a.length()) {
		if (a[i] == a[len]) {
			len++;
			arr[i] = len;
			i++;
		}
		else {
			if (len > 0) {
				len = arr[len - 1];
			}
			else {
				arr[i] = 0;
				i++;
			}
		}
	}
}

int lps(string s) {
	//mancher's algo.
	int c = 0, r = 0;
	int lps[s.size()] = {0};
	for (int i = 1; i < s.size() - 1; i++) {
		int mirr = c - (i - c);
		if (i < r) {
			lps[i] = min(lps[mirr], r - i);
		}
		while ((s[i + lps[i] + 1] != '@' ||
		        s[i + lps[i] + 1] != '@') &&
		        s[i + lps[i] + 1] == s[i - lps[i] - 1])lps[i]++;
		if (i + lps[i] > r) {
			c = i;
			r = i + lps[i];
		}
	}
}

void lcsonarray() {
	vi lcs(vi arr1, vi arr2) {
		int dp[arr1.size() + 1][arr2.size() + 1];
		memset(dp, 0, sizeof(dp));
		for (int i = 1; i <= arr1.size(); i++) {
			for (int j = 1; j <= arr2.size(); j++) {
				if (arr1[i - 1] == arr2[j - 1]) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
				}
				else {
					dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
				}
			}
		}
		vi ans;
		if (dp[arr1.size()][arr2.size()] == 0)return ans;
		int i = arr1.size(); int j = arr2.size();
		while (i > 0 && j > 0) {
			if (arr1[i - 1] == arr2[j - 1]) {
				ans.push_back(arr1[i - 1]);
				i--; j--;
			}
			else {
				if (dp[i][j - 1] > dp[i - 1][j]) {
					j--;
				}
				else {
					i--;
				}
			}
		}
		reverse(all(ans));
		return ans;
	}
}

void ExcelStrings() {
	queue<string>q;
	q.push("");		// as per will
	for (int i = 0; i < 17576; i++) {
		string t = q.front();
		q.pop();
		for (int j = 0; j < 26; j++) {
			t.pb('a' + j);
			q.push(t);
			if (!mp.count(t)) {
				cout << t << endl; return;
			}//print t
			t.pop_back();
		}
	}
}
void Substringsofdifferentlengths() {
	for (int i = 1; i <= s.size(); i++) {
		for (int j = 0; j < s.size() - i + 1; j++) {
		}
	}
}

int fact(int n) {
	int ans = 1;
	for (int i = 1; i <= n; i++)ans = ((ans % mod) * (i)) % mod;
	return (ans % mod);
}
int dig(int n) {
	return n == 0 ? 0 : (1 + dig(n / 10));
}
void median() {
	long long calcAnswer(vector <int> &numbers) {
		sort(all(numbers));
		long long result = 0;
		int sz = numbers.size();
		for (int i = 0; i < sz; ++i) result += abs(numbers[i] - numbers[sz / 2]);
		return result;
	}
}

void fibonacci_matrix_exponentiation() {
	void multiply(int F[2][2], int M[2][2]) {
		int a = F[0][0] * M[0][0] + F[0][1] * M[1][0];
		int b = F[0][0] * M[0][1] + F[0][1] * M[1][1];
		int c = F[1][0] * M[0][0] + F[1][1] * M[1][0];
		int d = F[1][0] * M[0][1] + F[1][1] * M[1][1];
		F[0][0] = a;
		F[0][1] = b;
		F[1][0] = c;
		F[1][1] = d;
	}
	void power(int F[2][2], int n) {
		if (n == 0 || n == 1)
			return;
		int M[2][2] = {{1, 1}, {1, 0}};
		power(F, n / 2);
		multiply(F, F);
		if (n % 2 != 0) {multiply(F, M);}
	}
	int find(int n) {
		int F[2][2] = {{1, 1}, {1, 0}};
		if (n == 0)
			return 0;
		power(F, n - 1);
		return F[0][0];
	}
}

void ncr() {
	long long modPow(long long a, long long p)
	{
		if (p == 0)
		{
			return 1;
		}
		if (p == 1)
		{
			return a;
		}

		long long res = modPow(a, p / 2);

		res = (res * res) % M;

		if (p & 1)
		{
			res = (a * res) % M;
		}

		return res;
	}
	int modPow(int a, int x, int p) {
		//calculates a^x mod p in logarithmic time.
		int res = 1;
		while (x > 0) {
			if ( x % 2 != 0) {
				res = (res * a) % p;
			}
			a = (a * a) % p;
			x /= 2;
		}
		return res;
	}
	int modInverse(int a, int p) {
		return modPow(a, p - 2, p);
	}
	int nCr(int n, int k, int p) {
		int numerator = 1;
		for (int i = 0; i < k; i++) {
			numerator = (numerator * (n - i) ) % p;
		}

		int denominator = 1;
		for (int i = 1; i <= k; i++) {
			denominator = (denominator * i) % p;
		}
		return ( numerator * modInverse(denominator, p) ) % p;
	}
	int addmod(int a, int b) {
		return ((a + b) % mod + mod) % mod;
	}
}

void nCr_optimal() {
	const int N = 300000 + 10;
	int factorialNumInverse[N + 1];

// array to precompute inverse of 1! to N!
	int naturalNumInverse[N + 1];

// array to store factorial of first N numbers
	int fact[N + 1];

// Function to precompute inverse of numbers
	void InverseofNumber(int p)
	{
		naturalNumInverse[0] = naturalNumInverse[1] = 1;
		for (int i = 2; i <= N; i++)
			naturalNumInverse[i] = naturalNumInverse[p % i] * (p - p / i) % p;
	}
// Function to precompute inverse of factorials
	void InverseofFactorial(int p)
	{
		factorialNumInverse[0] = factorialNumInverse[1] = 1;

		// precompute inverse of natural numbers
		for (int i = 2; i <= N; i++)
			factorialNumInverse[i] = (naturalNumInverse[i] * factorialNumInverse[i - 1]) % p;
	}

// Function to calculate factorial of 1 to N
	void factorial(int p)
	{
		fact[0] = 1;

		// precompute factorials
		for (int i = 1; i <= N; i++) {
			fact[i] = (fact[i - 1] * i) % p;
		}
	}

// Function to return nCr % p in O(1) time
	int Binomial(int N, int R, int p)
	{
		// n C r = n!*inverse(r!)*inverse((n-r)!)
		int ans = ((fact[N] * factorialNumInverse[R])
		           % p * factorialNumInverse[N - R])
		          % p;
		return ans;
	}
	int p = 998244353;
	InverseofNumber(p);
	InverseofFactorial(p);
	factorial(p);
}
void dfs(vi2 graph, int src, vi &ans, std::vector<bool> &visited) {
	visited[src] = true;
	ans.pb(src);
	for (int i : graph[src]) {
		if (visited[i] == false) {
			dfs(graph, i, ans, visited);
		}
	}
}
vb prime(200, true);
void sieve() {
	prime[1] = true;
	for (int i = 2; i < 200; i++) {
		if (prime[i]) {
			for (int j = i * 2; j < 200; j += i) {
				prime[j] = false;
			}
		}
	}
}
void primeandfactors() {
	bool prime[N];
	vector<int>primes;
	void  sieve_v2() {
		//prime numbers
		for (int i = 0; i < N; i++)
			prime[i] = 1;
		prime[0] = prime[1] = 0;
		for (int i = 4; i < N; i += 2) {
			prime[i] = 0;
		}
		for (int i = 3; i * i < N; i += 2) {
			if (prime[i]) {
				for (int j = i * i; j < N; j += i + i) {
					prime[j] = 0;
				}
			}
		}
		for (int i = 2; i < N; i++)
			if (prime[i])primes.push_back(i);
	}
	int factors(int x) {
		int ans = 0;
		for (int i = 0; primes[i]*primes[i] <= x; i++) {
			while (x % primes[i] == 0) {
				x /= primes[i];
				ans++;
			}
		}
		if (x > 1)ans++;
		return ans;
	}
}
void factors() {
	vi factors;
	void sieve(int b) {
		factors.pb(b);
		for (int i = 2; i * i <= b; i++) {
			if (b % i == 0) {
				factors.pb(i);
				if (i != (b / i))factors.pb(b / i);
			}
		}
	}
}
int find_rotated_array(const vector<int> &a, int b) {
	int n = a.size();
	int lo = 0;
	int hi = a.size() - 1;
	int ans = -1;
	while (lo <= hi) {
		int mid = lo + ((hi - lo) / 2);

		int next = (mid + 1) % n;
		int pre = (mid + n - 1) % n;

		if (a[mid] <= a[next] && a[mid] <= a[pre]) {
			ans = mid;
			return mid;
		}
		else if (a[0] <= a[mid]) {
			lo = mid + 1;
		}
		else if (a[mid] <= a[n - 1]) {
			hi = mid - 1;
		}
	}
	return ans;

}

void parsashumngoustree() {
	int p[N][2];
	int dp[N][2];
	vi graph[N];
	//travel till leaft and step back and start processing
	void dfs(int u, int par) {
		// dp[u][0] = dp[u][1] = 0;
		for (int x : graph[u]) {
			if (x == par)continue;
			dfs(x, u);//It will always take you to node before leaf
			cout << u << " " << x << " " << par << endl;
			int first = max(abs(p[u][0] - p[x][0]) + dp[x][0], abs(p[u][0] - p[x][1]) + dp[x][1]);
			dp[u][0] += first;
			int second = max(abs(p[u][1] - p[x][0]) + dp[x][0], abs(p[u][1] - p[x][1]) + dp[x][1]);
			dp[u][1] += second;
			// cout << dp[u][0] << " " << dp[u][1] << endl;
			cout << "calc" << endl;
		}
	}
}

void AVL() {
	class Node {
	public:
		int data = 0;
		Node* left = NULL;
		Node* right = NULL;
		int height = 0;
		int balance = 0;

		Node(int val) {
			data = val;
		}
		Node() {
			data = 0;
			left = NULL;
			right = NULL;
		}
	};

	void updateheightandbalance(Node * temp) {
		int lht = -1;
		int rht = -1;
		if (temp->left)lht = temp->left->height;
		if (temp->right)rht = temp->right->height;
		int balance = lht - rht;
		temp->height = max(lht, rht) + 1;
		temp->balance = balance;
	}

	Node* rightrotation(Node * root) {
		Node* rootleft = root->left;
		Node* rootleftleft = rootleft->right;
		rootleft->right = root;
		root->left = rootleftleft;
		updateheightandbalance(root);
		updateheightandbalance(rootleft);
		return rootleft;
	}
	Node* leftrotation(Node * root) {
		Node* rootright = root->right;
		Node* rootrightright = rootright->left;
		rootright->left = root;
		root->right = rootrightright;
		updateheightandbalance(root);
		updateheightandbalance(rootright);
		return rootright;
	}


	Node* getrotation(Node * root) {
		updateheightandbalance(root);
		if (root->balance == 2) {
			if (root->left->balance == 1) { //ll
				return rightrotation(root);
			}
			else { //lr
				root->left = leftrotation(root->left);
				return rightrotation(root);
			}
		}
		else if (root->balance == -2) {
			if (root->right->balance == -1) { //rr
				return leftrotation(root);
			}
			else { //rl
				root->right = rightrotation(root->right);
				return leftrotation(root);
			}
		}
		return root;
	}

	Node* insert(Node * root, int val) {
		if (root == NULL) {
			return new Node(val);
		}
		if (root->data > val) {
			root->left = insert(root->left, val);
		}
		else if (root->data < val) {
			root->right = insert(root->right, val);
		}
		root = getrotation(root);
		return root;
	}

	Node* lmax(Node * temp) {
		Node* t = temp;
		while (t->right != NULL)t = t->right;
		return t;
	}

	Node* remove(Node * root, int val) {
		if (root == NULL) {
			return NULL;
		}
		if (root->data > val) {
			root->left = insert(root->left, val);
		}
		else if (root->data < val) {
			root->right = insert(root->right, val);
		}
		else {
			cout << root->data << endl;
			if (root->left != NULL && root->right != NULL) {
				Node* l = lmax(root->left);
				root->data = l->data;
				root->left = remove(root->left, l->data);
				root = getrotation(root);
				return root;
				// return root.;
			}
			else if (root->left ) {
				return root->left;
			}
			else if (root->right ) {
				cout << root->data << endl;
				return root->right;
			}
			else if (!root->left && !root->right) {
				// cout << "HAHA" << endl;
				return NULL;
			}
		}
		return root;
	}

	void display(Node * root) {
		if (root == NULL) {return;}
		// cout << root->data << endl;
		if (root->left) {
			cout << root->left->data << " ";
		}
		else cout << "." << " ";
		cout << "<-" << " " << root->data << " " << "->" << " ";
		if (root->right) {
			cout << root->right->data << endl;
		}
		else cout << "." << endl;
		display(root->left);
		display(root->right);
	}
	//int main()
	vector<int>temp = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
	Node* root = NULL;
	// if (root == NULL)cout << "HAHA" << endl;
	for (int i : temp) {
		root = insert(root, i);
	}
	root = remove(root, 80);
	display(root);
}
void binarylifting() {
	// up[up[src][0]][0] parent ka parent
	int up[200001][20];
	void binary_lifting(int src, int par)
	{
		up[src][0] = par;
		for (int i = 1; i < 20; i++)
		{
			if (up[src][i - 1] != -1)
				up[src][i] = up[up[src][i - 1]][i - 1];
			else up[src][i] = -1;
		}

		for (int child : tree[src])
		{
			if (child != par)
				binary_lifting(child, src);
		}
	}
	int ans_query(int node, int jump_required)
	{
		if (node == -1 || jump_required == 0)
			return node;

		for (int i = 19; i >= 0; i--) {
			if (jump_required >= (1 << i)) {
				return ans_query(up[node][i], jump_required - (1 << i));
			}
		}
	}
	binary_lifting(1, -1);
	//undirected graph
	// searching kth node above given node in logn
}
void Trie() {
	struct trie {
		trie *child[26];
		int count;
		trie() {
			for (int i = 0; i < 26; i++) {
				child[i] = NULL;
			}
			count = 0;
		}
	};
	void insert(trie * root, string s) {
		trie* temp = root;
		for (int i = 0; i < s.size(); i++) {
			if (!temp->child[s[i] - 'a']) {
				temp->child[s[i] - 'a'] = new trie();
			}
			temp = temp->child[s[i] - 'a'];
			temp->count++;
		}
	}
	string search(trie * root, string s) {
		trie* temp = root;
		string ans;
		int n = s.size();
		for (int i = 0; i < n; i++) {
			if (temp->count == 1) {
				return ans;
			}
			ans.push_back(s[i]);
			temp = temp->child[s[i] - 'a'];
		}
		return s;
	}

	//while creating trie use Trie* root = new Trie();
	//trie is not limited to string and Maximum Xor Bit is evident for that
	//trie is mainly used for comparing prefixes ans above example did using bits
}
void DisjointSet() {
	//maintain in order clear globle map.
	class Node {
	public:
		Node* parent;
		int rank; int data;
	};
	unordered_map<int, Node*>mp;
	void make_set(int data) {
		Node* root = new Node();
		root->data = data;
		root->parent = root;
		root->rank = 0;
		mp[data] = root;
	}

	Node * find(Node * node) {
		if (node->parent == node)return node->parent;
		node->parent = find(node->parent);
		return node->parent;
	}
	void union_set(int data1, int data2) {
		Node* node1 = mp[data1];
		Node* node2 = mp[data2];

		Node* parent1 = find(node1);
		Node* parent2 = find(node2);
		if (parent1->data == parent2->data)return;
		if (parent1->rank >= parent2->rank) {
			if (parent1->rank == parent2->rank) {
				parent1->rank++;
			}
			parent2->parent = parent1;
		}
		else {
			parent1->parent = parent2;
		}
	}

	int find_set(int data) {
		return find(mp[data])->data;
	}
	bool compare(pair<int, pair<int, int>>a, pair<int, pair<int, int>>b) {
		return a.fir < b.fir;
	}

	//array implementation
	int parent[N], ran[N];
	void make_set(int v) {
		parent[v] = v;
		ran[v] = 0;
	}
	int find_set(int v) {
		if (v == parent[v])
			return v;
		return parent[v] = find_set(parent[v]);
	}
	void union_sets(int a, int b) {
		a = find_set(a);
		b = find_set(b);
		if (a != b) {
			if (ran[a] < ran[b])swap(a, b);
			parent[b] = a;
			if (ran[a] == ran[b])ran[a]++;
		}
	}
}

void SegmentTree() {
	void build(int node, int i, int j, vi arr, vi & s) {
		if (i >= j) {
			s[node] = arr[i];
			return;
		}
		int mid = (i + j) / 2;
		build(node * 2, i, mid, arr, s);
		build(node * 2 + 1, mid + 1, j, arr, s);
		s[node] = max(s[node * 2], s[node * 2 + 1]);
	}

	int query(int node, int st, int e, vi s, int l, int r) {
		if (e < l || st > r)return INT_MIN;
		if (st == e)return s[node];
		if (st == l && e == r)return s[node];
		int mid = (st + e) / 2;
		int left = query(node * 2, st, mid, s, l, r);
		int right = query(node * 2 + 1, mid + 1, e, s, l, r);
		return max(left, right);
	}
	vi s(4 * n, 0);
	build(1, 0, n - 1, arr, s);
	int ans = query(1, 0, n - 1, s, 1, 3);
}

void Aho_corasick() {
	class Node {
	public:
		unordered_map<char, Node*>child;
		Node* suffix_link;
		Node* output_link;
		int ind;
		Node() {
			suffix_link = NULL;
			output_link = NULL;
			ind = -1;
		}
	};

	void trie(Node * root, vector<string>arr) {
		for (int j = 0; j < arr.size(); j++) {
			Node* temp = root;
			string s = arr[j];
			for (int i = 0; i < s.size(); i++) {
				if (temp->child.count(s[i]))temp = temp->child[s[i]];
				else {
					temp->child[s[i]] = new Node();
					temp = temp->child[s[i]];
				}
			}
			temp->ind = j;
		}

	}

	void Aho_Corasick_BFS(Node * root) {
		root->suffix_link = root;
		queue<Node*>q;
		// q.push(root);
		for (auto it : root->child) {
			root->child[it.fir]->suffix_link = root;
			q.push(it.sec);
		}
		while (q.size()) {
			auto front = q.front();
			q.pop();
			for (auto it : front->child) {
				Node* t = front->suffix_link;
				while (!t->child[it.fir] && t != root)t = t->suffix_link;
				if (t->child.count(it.fir)) {front->child[it.fir]->suffix_link = t->child[it.fir];}
				else it.sec->suffix_link = root;
				q.push(it.sec);
			}
			if (front->suffix_link->ind != -1) {
				front->output_link = front->suffix_link;
			}
			else {
				front->output_link = front->suffix_link->output_link;
			}
		}

	}

	void query(Node * root, string s, vi2 & arr) {
		Node* t = root;
		for (int i = 0; i < s.size(); i++) {
			if (t->child.count(s[i])) {
				t = t->child[s[i]];
				if (t->ind != -1) {
					arr[t->ind].pb(i);
				}
				Node* m = t->output_link;
				while (m != NULL) {
					arr[m->ind].pb(i);
					m = m->output_link;
				}
				// t = t->child[s[i]];
			}
			else {
				while (t != root && !t->child.count(s[i]))t = t->suffix_link;
				if (t->child.count(s[i]))i--;
				if (i + 1 == s.size() - 1)i++;
			}
		}
	}
	int main() {
		vector<string>s = {"ACC", "ATC", "CAT", "GCG", "A", "T", "C"};
		Node* root = new Node();
		trie(root, s);
		Aho_Corasick_BFS(root);
		vi2 arr(s.size());
		query(root, "GCATCG", arr);
		// cout << ((root->child['G']->child['C']->child.count('A')) ? "HHA" : "NNA");
		for (int i = 0; i < arr.size(); i++) {
			if (arr[i].size() == 0)cout << -1 << endl;
			else {
				for (int j = 0; j < arr[i].size(); j++) {
					cout << arr[i][j] - s[i].size() + 1 << " " ;
				}
				cout << endl;
			}
		}
	}
}


void getTheDistanceBetweenAnyTwoNodesOfATreeInLOG(N) {
	vector<int> ar[100001];
	const int maxN = 1000; //power of two
	int level[100001] , LCA[100001][maxN + 1];


	void dfs(int node , int lvl , int par)
	{
		level[node] = lvl;
		LCA[node][0] = par;

		for (int child : ar[node])
			if (child != par)
			{
				dfs(child , lvl + 1 , node);
			}
	}


	void init(int n)
	{
		dfs(1 , 0 , -1);

		for (int i = 1; i <= maxN; i++)
		{
			for (int j = 1; j <= n; j++)
				if (LCA[j][i - 1] != -1)
				{
					int par = LCA[j][i - 1];
					LCA[j][i] = LCA[par][i - 1];
				}
		}
	}

	int getLCA(int a , int b)
	{
		if (level[b] < level[a]) swap(a , b);

		int d = level[b] - level[a];

		while (d > 0)
		{
			//optimization
			int i = log2(d);
			b = LCA[b][i];

			d -= 1 << i;
		}

		if (a == b) return a;

		for (int i = maxN; i >= 0; i--)
			if (LCA[a][i] != -1 && (LCA[a][i] != LCA[b][i]))
			{
				a = LCA[a][i] , b = LCA[b][i];
			}

		return LCA[a][0];
	}


	int getDist(int a , int b)
	{
		int lca = getLCA(a , b);
		return level[a] + level[b] - 2 * level[lca];
	}
}


void Binary_Search_Template() {
	int i = 1; int j = n;
	while (i < j) {
		int mid = (i + j) / 2;
		int t = query(mid, n);
		if (mid == i)break;
		if (t < a) {
			j = mid;
		}
		else {
			i = mid;
		}
	}
}


void bitmanipulation() {
	int topbit(signed t) {
		return t == 0 ? -1 : 31 - __builtin_clz(t);
	}
	int topbit(ll t) {
		return t == 0 ? -1 : 63 - __builtin_clzll(t);
	}
	int botbit(signed a) {
		return a == 0 ? 32 : __builtin_ctz(a);
	}
	int botbit(int a) {
		return a == 0 ? 64 : __builtin_ctzll(a);
	}
	int popcount(signed t) {
		return __builtin_popcount(t);
	}
	int popcount(ll t) {
		return __builtin_popcountll(t);
	}
	bool ispow2(int i) {
		return i && (i & -i) == i;
	}
	ll mask(int i) {
		return (ll(1) << i) - 1;
	}
}
void LocalPartitioninarray() {
	vector<int>vec(n);
	for (int i = 0; i < n; i++) cin >> vec[i];
	vector<int>mins(n);
	mins[n - 1] = vec[n - 1];
	for (int i = n - 2; i >= 0; i--) {
		mins[i] = min(mins[i + 1], vec[i]);
	}
	int ans = 0;
	int mn = INT_MAX, mx = INT_MIN;
	for (int i = 0; i < n; i++) {
		mn = min(mn, vec[i]);
		mx = max(mx, vec[i]);
		if (i + 1 == n || mx <= mins[i + 1]) {
			ans += mx - mn;
			mn = INT_MAX;
			mx = INT_MIN;
		}

	}
	cout << ans << "\n";
}
vector<int> decimalToBinary(long long number) {
	//10 will become -> 00000000000001010
	vector<int> binary;
	while (number) {
		binary.push_back(number % 2);
		number /= 2;
	}
	int left = 18 - binary.size();
	while (left--) binary.push_back(0);
	reverse(binary.begin(), binary.end());
	return binary;
}
int LongestIncreasingSubsequenceLengthNLOGN(std::vector<int>& v)
{
	if (v.size() == 0) // boundary case
		return 0;

	std::vector<int> tail(v.size(), 0);
	int length = 1; // always points empty slot in tail

	tail[0] = v[0];

	for (int i = 1; i < v.size(); i++) {

		// Do binary search for the element in
		// the range from begin to begin + length
		auto b = tail.begin(), e = tail.begin() + length;
		auto it = upper_bound(b, e, v[i]);

		// If not present change the tail element to v[i]
		if (it == tail.begin() + length)
			tail[length++] = v[i];
		else
			*it = v[i];
	}

	return length;
}