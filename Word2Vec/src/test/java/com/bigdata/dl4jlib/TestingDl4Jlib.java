package com.bigdata.dl4jlib;

import java.io.IOException;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

/**
 * Class testing Deep Learning For Java Library<br>
 * Using Word2vec<br>
 * For more information about lib: http://deeplearning4j.org/word2vec.html#anatomy
 * 
 * @created 11 / 7 / 2015
 * @author baonc
 *
 */
public class TestingDl4Jlib {
	public static void main(String args[]) throws IOException {
		int batchSize = 1000;
		int iterations = 1;
		int layerSize = 300;
		
		Nd4j.getRandom().setSeed(133);
		System.out.println("Load data...");
		ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
		SentenceIterator iter = new LineSentenceIterator(resource.getFile());
		iter.setPreProcessor(new SentencePreProcessor() {
			private static final long serialVersionUID = 1L;

			@Override
			public String preProcess(String sentence) {
				return sentence.toLowerCase();
			}
		});
		
		System.out.println("Tokenizer data...");
		final EndingPreProcessor preProcessor = new EndingPreProcessor();
		TokenizerFactory tokenizer = new DefaultTokenizerFactory();
		tokenizer.setTokenPreProcessor(new TokenPreProcess() {
			
			@Override
			public String preProcess(String token) {
				token = token.toLowerCase();
				String base = preProcessor.preProcess(token);
				base = base.replaceAll("\\d", "d");
				if(base.endsWith("ly") || base.endsWith("ing")) {
					System.out.println();
				}
				return base;
			}
		});
		
		System.out.println("Build model...");
		Word2Vec vec = new Word2Vec.Builder().batchSize(batchSize).sampling(1e-5)
				.minWordFrequency(5).useAdaGrad(false).layerSize(layerSize).iterations(iterations)
				.learningRate(0.025).minLearningRate(1e-2).negativeSample(0).iterate(iter)
				.tokenizerFactory(tokenizer).build();
		vec.fit();
		
		InMemoryLookupTable table = (InMemoryLookupTable)vec.lookupTable();
		table.getSyn0().diviRowVector(table.getSyn0().norm2(0));
		
		System.out.println("Evaluate model...");
		double sim = vec.similarity("people", "money");
		System.out.println("Similarity between people and money: " + sim);
		Collection<String> similar = vec.wordsNearest("day", 20);
		System.out.println("Top 20 word nearest with day:");
		similar.forEach(simila -> System.out.print(simila + " "));
	}
}