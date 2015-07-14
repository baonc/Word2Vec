package com.bigdata.vccorp.word2vec;

import java.util.Random;

import com.bigdata.vccorp.processdata.BuildVocabulary;
import com.bigdata.vccorp.processdata.OneHotEncode;

/**
 * Class process one-word context in word2vec model<br>
 * For more information see: Xin Rong,. word2vec Parameter Learning Explained
 * 
 * @created 13 / 7 / 2015
 * @author baonc
 * @github https://github.com/baonc/Word2Vec
 */
public class OneWordContext {
	private static final int SIZE_OF_HIDDEN_LAYER = 3;	// dimension of hidden layer
	private static final double LEARNING_RATE = 1;		// learning rate of neural network
	
	private BuildVocabulary buildVocabulary;
	private OneHotEncode oneHotEncode;
	
	private int sizeOfDic;								// size of vocabulary
	private double w1[][];								// w matrix
	private double w2[][];								// w' matix
	private double h[];									// hidden layer h = {x'}^TW
	private double u[];									// score for each word uj = {v'}_{w_j}^Th
	private double y[];									// output of the j-the node in the output
														// layer
	private double e[];
	private double eh[];
	
	/**
	 * Constructor<br>
	 * Initialize parameter
	 */
	public OneWordContext() {
		buildVocabulary = new BuildVocabulary();
		buildVocabulary.buildVocabulary();
		sizeOfDic = buildVocabulary.getVocabularySize();
		
		oneHotEncode = new OneHotEncode();
		
		w1 = new double[sizeOfDic][SIZE_OF_HIDDEN_LAYER];
		w2 = new double[SIZE_OF_HIDDEN_LAYER][sizeOfDic];
		h = new double[SIZE_OF_HIDDEN_LAYER];
		u = new double[sizeOfDic];
		y = new double[sizeOfDic];
		e = new double[sizeOfDic];
		eh = new double[SIZE_OF_HIDDEN_LAYER];
	}
	
	/**
	 * Get W matrix
	 * 
	 * @return	: W matrix of neural network
	 */
	public double[][] getW1() {
		return w1;
	}
	
	/**
	 * Get  W' matrix
	 * 
	 * @return	: W' matrix of neural network
	 */
	public double[][] getW2() {
		return w2;
	}
	
	/**
	 * Function calculating sof-max of word i in vocabulary
	 * 
	 * @param i	: ith word
	 * @return	: softMax of this word
	 */
	public double softMaxFunction(int i) {
		double denominator = 0;
		for(int j = 0; j < sizeOfDic; j++) {
			denominator += Math.exp(u[j]);
		}
		
		double numerator = Math.exp(u[i]);
		
		return numerator / denominator;
	}
	
	/**
	 * Transport of neural network.
	 */
	public void transport(String inputWord) {
		int encodeInputWord[] = oneHotEncode.oneHot(inputWord);
		
		// from input layer to hidden layer
		int k = 0;
		for(int i = 0; i < encodeInputWord.length; i++) {
			if(encodeInputWord[i] == 1) {
				k = i;
			}
		}
		
		h = w1[k];	// update h.
		
		// from hidden layer to output layer
		for(int i = 0; i < u.length; i++) {
			u[i] = 0;
			for(int j = 0; j < SIZE_OF_HIDDEN_LAYER; j++) {
				u[i] += w2[j][i] * h[j];
			}
		}
		
		// posterior distribution of words
		for(int i = 0; i < y.length; i++) {
			y[i] = softMaxFunction(i);
		}
	}
	
	/**
	 * Function initialize for W and W' matrix
	 */
	public void initWmatrix() {
		Random r = new Random();
		
		for(int i = 0; i < w1.length; i++) {
			for(int j = 0; j < SIZE_OF_HIDDEN_LAYER; j++) {
				w1[i][j] = r.nextDouble();
				w2[j][i] =r.nextDouble();
			}
		}
	}
	
	/**
	 * Update equation for hidden→output weights<br>
	 * Update W' function.<br>
	 * With: <br>
	 * {w'}_{ij}^{(new)} = {w'}_{ij}^{(old)} - \eta e_jh_i<br>
	 * For more information see: Xin Rong,. word2vec Parameter Learning Explained
	 * 
	 * @param inputWord		: input of neural network
	 * @param targetWord	: output of neural network
	 */
	public void updateW2(String inputWord, String targetWord) {
		int encodeOfTargetWord[] = oneHotEncode.oneHot(targetWord);
		
		// first transport in neural network with input word
		transport(inputWord);
		
		// calculating e.
		for(int i = 0; i < e.length; i++) {
			if(encodeOfTargetWord[i] == 1) {
				e[i] = y[i] - 1;
			} else {
				e[i] = y[i];
			}
		}
		
		// update W'
		for(int i = 0; i < SIZE_OF_HIDDEN_LAYER; i++) {
			for(int j = 0; j < sizeOfDic; j++) {
				w2[i][j] = w2[i][j] - LEARNING_RATE * e[j] * h[i];
			}
		}
	}
	
	/**
	 * Update equation for input→hidden weights<br>
	 * After update W', we need to update W.<br>
	 * For more information see: Xin Rong,. word2vec Parameter Learning Explained
	 */
	public void updateW1(String inputWord) {
		int[] encodeOfInputWord = oneHotEncode.oneHot(inputWord);
		
		 // calculating eh
		for(int i = 0 ; i < SIZE_OF_HIDDEN_LAYER; i++) {
			eh[i] = 0;
			for(int j = 0; j < sizeOfDic; j++) {
				eh[i] += e[j] * w2[i][j];
			}
		}
		
		// update W
		for(int i = 0; i < sizeOfDic; i++) {
			for(int j = 0; j < SIZE_OF_HIDDEN_LAYER; j++) {
				w1[i][j] = w1[i][j] - LEARNING_RATE * eh[j] * encodeOfInputWord[i];
			}
		}
	}
	
	/**
	 * Learn the neural network
	 * 
	 * @param inputWord		: input work
	 * @param targetWord	: output work
	 */
	public void learnNetwork(String inputWord, String targetWord) {
		// first initialize W and W'
		initWmatrix();
		// update W'
		updateW2(inputWord, targetWord);
		// update W
		updateW1(inputWord);
	}
}