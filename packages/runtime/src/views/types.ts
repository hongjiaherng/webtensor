import { Node } from '@webtensor/ir';
import { RuntimeTensor } from '../backend';

/**
 * A view function creates a zero-copy RuntimeTensor by recomputing
 * shape/strides/offset metadata without allocating new storage.
 */
export type ViewFn = (node: Node, src: RuntimeTensor) => RuntimeTensor;
