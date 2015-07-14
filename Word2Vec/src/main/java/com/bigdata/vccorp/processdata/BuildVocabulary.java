package com.bigdata.vccorp.processdata;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;

/**
 * Class build a vocabulary from input file<br>
 * 
 * @created 13 / 7 / 2015
 * @author baonc
 * @github https://github.com/baonc/Word2Vec
 */
public class BuildVocabulary {
	private HashSet<String> vocabulary;
	private static final String INPUT_PATH = "src/main/resources/smallexample.txt";
	
	/**
	 * Constructor
	 */
	public BuildVocabulary() {
		vocabulary = new HashSet<>();
	}
	
	/**
	 * Read data from INPUT_PATH and set vocabulary
	 */
	public void buildVocabulary() {
		try(InputStream in = Files.newInputStream(Paths.get(INPUT_PATH));
				BufferedReader reader = new BufferedReader(new InputStreamReader(in))) {
			String line;
			while((line = reader.readLine()) != null) {
				String words[] = line.split(" ");
				for(int i = 0; i < words.length; i++) {
					vocabulary.add(words[i]);
				}
			}
		} catch(IOException ioe) {
			ioe.printStackTrace();
		}
	}
	
	/**
	 * Get vocabulary
	 * 
	 * @return	: vocabulary
	 */
	public HashSet<String> getVocabulary() {
		return vocabulary;
	}
	
	/**
	 * Get size of vocabulary
	 * 
	 * @return	: size of vocabulary
	 */
	public int vocabularySize() {
		return vocabulary.size();
	}
	
	/**
	 * Test function
	 * 
	 * @param args	: main args
	 */
	public static void main(String args[]) {
		BuildVocabulary voca = new BuildVocabulary();
		voca.buildVocabulary();
		HashSet<String> vocabularies = voca.getVocabulary();
		vocabularies.forEach(vocabulary -> System.out.println(vocabulary));
	}
}