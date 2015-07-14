package org.bigdata.word2vec;

import com.bigdata.vccorp.word2vec.OneWordContext;

/**
 * Class test one-word context model
 * 
 * @created 14 / 7 / 2015
 * @author baonc
 * @github https://github.com/baonc/Word2Vec
 */
public class TestOneWordContext {
	public static void main(String args[]) {
		double w1[][];
		double w2[][];
		OneWordContext oneWordModel = new OneWordContext();
		
		w1 = oneWordModel.getW1();
		w2 = oneWordModel.getW2();
		
		System.out.println("Before training: ");
		System.out.println("W:");
		for(int i = 0; i < w1.length; i++) {
			for(int j = 0; j < w1[i].length; j++) {
				System.out.print(w1[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println("W':");
		for(int i = 0; i < w2.length; i++) {
			for(int j = 0; j < w2[i].length; j++) {
				System.out.print(w2[i][j] + " ");
			}
			System.out.println();
		}

		oneWordModel.learnNetwork("king", "queen");
		w1 = oneWordModel.getW1();
		w2 = oneWordModel.getW2();
		
		System.out.println("After training: ");
		System.out.println("W:");
		for(int i = 0; i < w1.length; i++) {
			for(int j = 0; j < w1[i].length; j++) {
				System.out.print(w1[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println("W':");
		for(int i = 0; i < w2.length; i++) {
			for(int j = 0; j < w2[i].length; j++) {
				System.out.print(w2[i][j] + " ");
			}
			System.out.println();
		}
	}
}
