import java.io.BufferedInputStream
import java.io.InputStream
import java.util.StringTokenizer


const val MOD = 123
const val YES = "yes"
const val NO = "NO"

fun solve(N: Long, M: Long, H: Array<Array<String>>, A: LongArray, B: DoubleArray, Q: Long, X: LongArray){
    println("$N $M")
    assert (H.size == (N - 1).toInt())
    for (i in 0 until (N - 1).toInt()) {
        assert(H[i].size == (M - 2).toInt())
        for (j in 0 until (M - 2).toInt()) {
            if (j >0){ print(" ") }
            print(H[i][j])
        }
        println()
    }
    assert (A.size == (N - 1).toInt())
    assert (B.size == (N - 1).toInt())
    for (i in 0 until (N - 1).toInt()) {
        println("" + A[i] + " " + B[i]);
    }
    println(Q)
    assert (X.size == (M + Q).toInt())
    for (i in 0 until (M + Q).toInt()) {
        println(X[i])
    }
    println(YES)
    println(NO)
    println(MOD)
    return
}

// Generated by x.y.z https://github.com/kyuridenamida/atcoder-tools  (tips: You use the default template now. You can remove this line by using your custom template)
fun main(args: Array<String>) {
    fun StringArray(size: Int, init: (Int) -> String = { "\u0000" }): Array<String> {
        return Array<String>(size, init)
    }
    class Scanner(stream: InputStream) {
        private val reader = BufferedInputStream(stream).bufferedReader()
        private var st: StringTokenizer? = null
        fun next(): String {
            while (!(st?.hasMoreTokens() ?: false)) st = StringTokenizer(reader.readLine())
            return st!!.nextToken()
        }
    }
    val sc = Scanner(System.`in`)
    val N = sc.next().toLong()
    val M = sc.next().toLong()
    val H = Array<Array<String>>((N-2+1).toInt()){ StringArray((M-1-2+1).toInt()) }
    for (i in 0 until (N-2+1).toInt()) {
        for (j in 0 until (M-1-2+1).toInt()) {
            H[i][j] = sc.next()
        }
    }
    val A = LongArray((N-2+1).toInt())
    val B = DoubleArray((N-2+1).toInt())
    for (i in 0 until (N-2+1).toInt()) {
        A[i] = sc.next().toLong()
        B[i] = sc.next().toDouble()
    }
    val Q = sc.next().toLong()
    val X = LongArray((M+Q).toInt())
    for (i in 0 until (M+Q).toInt()) {
        X[i] = sc.next().toLong()
    }
    solve(N, M, H, A, B, Q, X)
}
