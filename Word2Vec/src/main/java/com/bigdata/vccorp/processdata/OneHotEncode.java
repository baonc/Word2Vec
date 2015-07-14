package com.bigdata.vccorp.processdata;

import java.util.HashSet;

/**
 * Class encode a word with one-hot<br>
 * meaning they are vectors of length V (the size of the vocabulary) with a value of 1 at the
 *  index corresponding to the word and zeros in all other indexes
 * 
 * @created 14 / 7 / 2015
 * @author baonc
 * @github: https://github.com/baonc/Word2Vec
 *
 */
public class OneHotEncode {
	private HashSet<String> vocabulary;
	private BuildVocabulary buildVocabulary;
	
	/**
	 * Constructor	: init vocabulary
	 */
	public OneHotEncode() {
		buildVocabulary = new BuildVocabulary();
		buildVocabulary.buildVocabulary();
		vocabulary = buildVocabulary.getVocabulary();
	}
	
	/**
	 * One-hot one word with vocabulary.
	 * 
	 * @param word	: word
	 * @return		: one-hot encode of the word
	 */
	public int[] oneHot(String word) {
		int oneHot[] = new int[vocabulary.size()];
		
		int index = 0;
		for(String voca : vocabulary) {
			if(!voca.equals(word)) {
				index++;
			} else {
				break;
			}
		}
		
		oneHot[index] = 1;
		
		return oneHot;
	}

	/**
	 * Test function
	 * 
	 * @param args	: main args
	 */
	public static void main(String args[]) {
		OneHotEncode oneHot = new OneHotEncode();
		int encode[] = oneHot.oneHot("loves");
		for(int i = 0; i < encode.length; i++) {
			System.out.print(encode[i] + " ");
		}
	}
}